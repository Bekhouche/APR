import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
import scann
import os
import pickle
from typing import List, Tuple

class AttentiveRelevanceScoring(nn.Module):
     def __init__(self, embedding_dim, attention_hidden_dim=None, reg_lambda=0.1):
          super().__init__()
          self.embedding_dim = embedding_dim
          self.attention_hidden_dim = attention_hidden_dim or embedding_dim
          self.reg_lambda = reg_lambda

          self.query_proj = nn.Linear(embedding_dim, self.attention_hidden_dim)
          self.ctx_proj = nn.Linear(embedding_dim, self.attention_hidden_dim)

          self.attn_vector = nn.Parameter(torch.randn(self.attention_hidden_dim))
          self.bias = nn.Parameter(torch.tensor(0.0))

          self.temperature = nn.Parameter(torch.tensor(1.0))

     def forward(self, query_embeddings, ctx_embeddings, is_train=True):
          B = query_embeddings.size(0)
          eps = 1e-8

          if is_train:
               assert ctx_embeddings.size(0) == 2 * B, "ctx_embeddings must be double the query_embeddings during training"

               ctx_pos = ctx_embeddings[:B]
               ctx_neg = ctx_embeddings[B:]

               h_q = self.query_proj(query_embeddings)        # [B, H]
               h_p_pos = self.ctx_proj(ctx_pos)               # [B, H]
               h_p_neg = self.ctx_proj(ctx_neg)               # [B, H]

               a_pos = torch.tanh(h_q * h_p_pos)              # [B, H]
               a_neg = torch.tanh(h_q * h_p_neg)              # [B, H]

               sim_pos = torch.matmul(a_pos, self.attn_vector)  # [B]
               sim_neg = torch.matmul(a_neg, self.attn_vector)  # [B]

               s_pos = (sim_pos + self.bias) / self.temperature
               s_neg = (sim_neg + self.bias) / self.temperature

               r_pos = torch.sigmoid(s_pos)
               r_neg = torch.sigmoid(s_neg)

               loss = -torch.log(r_pos + eps).mean() - torch.log(1 - r_neg + eps).mean()
               loss += self.reg_lambda * (r_pos.std() + r_neg.std()) 
               loss += self.reg_lambda * (s_pos.std() + s_neg.std()) 

               scores = torch.cat([r_pos, r_neg], dim=0)
               return scores, loss

          else:
               num_ctx = ctx_embeddings.size(0)
               h_q = self.query_proj(query_embeddings)         # [B, H]
               h_p = self.ctx_proj(ctx_embeddings)             # [num_ctx, H]

               if B == 1:
                    h_q = h_q.expand(num_ctx, -1)
               elif B != num_ctx:
                    raise ValueError(f"In inference mode, query batch size ({B}) must be 1 or match ctx batch size ({num_ctx})")

               a = torch.tanh(h_q * h_p)                        # [num_ctx, H]
               sim_scores = torch.matmul(a, self.attn_vector)  # [num_ctx]

               s = (sim_scores + self.bias) / self.temperature
               r = torch.sigmoid(s)
               return r, None

class ScaNNDenseIndexer:
     def __init__(self, num_leaves: int = 2000, num_leaves_to_search: int = 100, training_sample_size: int = 250000):
          self.index = None
          self.db_vectors = None
          self.db_ids = []
          self.num_leaves = num_leaves
          self.num_leaves_to_search = num_leaves_to_search
          self.training_sample_size = training_sample_size
          self.scann_searcher = None

     def _normalize(self, vectors):
          return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

     def index_data(self, data: List[Tuple[object, np.ndarray]]):
          self.db_ids = [x[0] for x in data]
          self.db_vectors = np.stack([x[1] for x in data])
          self.db_vectors = self._normalize(self.db_vectors)

          num_points = len(self.db_vectors)
          
          effective_num_leaves = min(self.num_leaves, max(1, num_points))
          
          effective_num_leaves_to_search = min(self.num_leaves_to_search, effective_num_leaves)
          
          effective_training_sample_size = min(self.training_sample_size, num_points)
          
          if num_points < 100:
               self.scann_searcher = scann.scann_ops_pybind.builder(self.db_vectors, 10, "dot_product") \
                    .tree(num_leaves=effective_num_leaves, num_leaves_to_search=effective_num_leaves_to_search, training_sample_size=effective_training_sample_size) \
                    .score_brute_force() \
                    .build()
          else:
               self.scann_searcher = scann.scann_ops_pybind.builder(self.db_vectors, 10, "dot_product") \
                    .tree(num_leaves=effective_num_leaves, num_leaves_to_search=effective_num_leaves_to_search, training_sample_size=effective_training_sample_size) \
                    .score_ah(2, anisotropic_quantization_threshold=0.2) \
                    .reorder(min(100, num_points)) \
                    .build()

     def search_knn(self, query_vectors: np.ndarray, top_k: int = 10):
          query_vectors = self._normalize(query_vectors)
          neighbors = [self.scann_searcher.search_batched(query_vec, final_num_neighbors=top_k)
                         for query_vec in query_vectors]

          results = []
          for dists, indices in neighbors:
               docs = [self.db_ids[i] for i in indices]
               results.append((docs, dists.tolist()))
          return results

     def serialize(self, path: str):
          os.makedirs(path, exist_ok=True)
          with open(os.path.join(path, "db_ids.pkl"), "wb") as f:
               pickle.dump(self.db_ids, f)
          np.save(os.path.join(path, "db_vectors.npy"), self.db_vectors)

     def deserialize(self, path: str):
          with open(os.path.join(path, "db_ids.pkl"), "rb") as f:
               self.db_ids = pickle.load(f)
          self.db_vectors = np.load(os.path.join(path, "db_vectors.npy"))
          
          num_points = len(self.db_vectors)
          effective_num_leaves = min(self.num_leaves, max(1, num_points))
          effective_num_leaves_to_search = min(self.num_leaves_to_search, effective_num_leaves)
          effective_training_sample_size = min(self.training_sample_size, num_points)
          
          if num_points < 100:
               self.scann_searcher = scann.scann_ops_pybind.builder(self.db_vectors, 10, "dot_product") \
                    .tree(effective_num_leaves, effective_num_leaves_to_search, effective_training_sample_size) \
                    .score_brute_force() \
                    .build()
          else:
               self.scann_searcher = scann.scann_ops_pybind.builder(self.db_vectors, 10, "dot_product") \
                    .tree(effective_num_leaves, effective_num_leaves_to_search, effective_training_sample_size) \
                    .score_ah(2, anisotropic_quantization_threshold=0.2) \
                    .reorder(min(100, num_points)) \
                    .build()

class DynamicDPR(nn.Module):
     def __init__(self, model_name):
          super().__init__()

          self.question_encoder = AutoModel.from_pretrained(
               model_name,
               output_hidden_states=True,
               return_dict=True
          )

          self.ctx_encoder = AutoModel.from_pretrained(
               model_name,
               output_hidden_states=True,
               return_dict=True
          )
          
          self.max_length = 256
          
          self.relevance_scoring = AttentiveRelevanceScoring(
               embedding_dim=self.question_encoder.config.hidden_size
          )
          
          self.indexer = ScaNNDenseIndexer()
     
     def create_index(self, documents: List[Tuple[str, str]], batch_size: int = 32):
          """Create a searchable index from a list of documents.
          
          Args:
               documents: List of (doc_id, text) tuples
               batch_size: Batch size for encoding documents
          """
          self.eval()
          all_embeddings = []
          all_ids = []
          
          with torch.no_grad():
               for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    texts = [doc[1] for doc in batch]
                    ids = [doc[0] for doc in batch]
                    
                    # Tokenize and encode
                    inputs = self.tokenizer(
                         texts,
                         max_length=self.max_length,
                         padding='max_length',
                         truncation=True,
                         return_tensors='pt'
                    ).to(next(self.parameters()).device)
                    
                    outputs = self.ctx_encoder(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_ids.extend(ids)
          
          # Create index
          data = list(zip(all_ids, np.vstack(all_embeddings)))
          self.indexer.index_data(data)
     
     def search(self, query: str, top_k: int = 10):
          """Search for relevant documents using the query.
          
          Args:
               query: Query text
               top_k: Number of results to return
               
          Returns:
               List of (doc_id, score) tuples
          """
          self.eval()
          with torch.no_grad():
               # Encode query
               inputs = self.tokenizer(
                    query,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
               ).to(next(self.parameters()).device)
               
               outputs = self.question_encoder(**inputs)
               query_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
               query_embedding = F.normalize(query_embedding, p=2, dim=1)
               
               results = self.indexer.search_knn(query_embedding.cpu().numpy(), top_k=top_k)
               return results[0]  # Return first batch result

     def forward(self, question_inputs, ctx_inputs, is_train=True):
          question_outputs = self.question_encoder(**question_inputs)
          question_embedding = question_outputs.last_hidden_state[:, 0, :]  # [CLS] token
          
          ctx_outputs = self.ctx_encoder(**ctx_inputs)
          ctx_embedding = ctx_outputs.last_hidden_state[:, 0, :]  # [CLS] token
          
          question_embedding = F.normalize(question_embedding, p=2, dim=1)
          ctx_embedding = F.normalize(ctx_embedding, p=2, dim=1)
          
          scores, dynamic_loss = self.relevance_scoring(question_embedding, ctx_embedding, is_train)
          
          return question_embedding, ctx_embedding, scores, dynamic_loss
