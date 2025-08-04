"""Latent action encoding system for bandwidth-efficient drone coordination."""

import numpy as np
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class CompressionType(Enum):
    """Compression algorithm types."""
    LEARNED_VQVAE = "learned_vqvae"
    NEURAL_COMPRESSION = "neural_compression"
    DICTIONARY_COMPRESSION = "dictionary_compression"
    SIMPLE_QUANTIZATION = "simple_quantization"


@dataclass
class EncodingMetrics:
    """Encoding performance metrics."""
    compression_ratio: float = 0.0
    encoding_time_ms: float = 0.0
    decoding_time_ms: float = 0.0
    reconstruction_error: float = 0.0
    bandwidth_saved_percent: float = 0.0


class SimpleVQVAE(nn.Module):
    """Simplified Vector Quantized Variational Autoencoder for action encoding."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Vector quantization layer
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VQ-VAE."""
        # Encode
        z_e = self.encoder(x)
        
        # Vector quantization
        z_q, indices = self.vector_quantize(z_e)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, z_e, z_q, indices
    
    def vector_quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vector quantization operation."""
        # Calculate distances to embedding vectors
        distances = torch.cdist(z_e, self.embedding.weight)
        
        # Find closest embeddings
        indices = torch.argmin(distances, dim=-1)
        
        # Get quantized vectors
        z_q = self.embedding(indices)
        
        return z_q, indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to quantized latent representation."""
        z_e = self.encoder(x)
        _, indices = self.vector_quantize(z_e)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode quantized indices back to original space."""
        z_q = self.embedding(indices)
        return self.decoder(z_q)


class LatentEncoder:
    """Latent action encoder for bandwidth-efficient drone communication.
    
    Compresses high-dimensional action sequences into compact latent codes
    using various compression algorithms including learned representations.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        latent_dim: int = 512,
        compression_type: str = "learned_vqvae",
        device: str = "cpu"
    ):
        """Initialize latent encoder.
        
        Args:
            input_dim: Input dimension (e.g., LLM embedding size)
            latent_dim: Compressed latent dimension
            compression_type: Compression algorithm to use
            device: Computing device (cpu/cuda)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.compression_type = CompressionType(compression_type)
        self.device = device
        
        # Initialize compression model
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
        # Performance metrics
        self.metrics = EncodingMetrics()
        self.encoding_history: List[Dict[str, Any]] = []
        
        # Emergency/control codes
        self.emergency_codes = {
            'stop': np.array([999] * min(64, latent_dim)),
            'land': np.array([998] * min(64, latent_dim)),
            'return_home': np.array([997] * min(64, latent_dim)),
            'hold_position': np.array([996] * min(64, latent_dim)),
        }
        
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the compression model based on type."""
        if self.compression_type == CompressionType.LEARNED_VQVAE:
            self.model = SimpleVQVAE(
                input_dim=self.input_dim,
                latent_dim=64,  # Use smaller dim for indices
                num_embeddings=512
            ).to(self.device)
            
        elif self.compression_type == CompressionType.NEURAL_COMPRESSION:
            # Simple autoencoder for neural compression
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.latent_dim),
                nn.Tanh()  # Normalized output
            ).to(self.device)
            
        elif self.compression_type == CompressionType.DICTIONARY_COMPRESSION:
            # Initialize with pre-trained text encoder for dictionary-based compression
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.model = AutoModel.from_pretrained("microsoft/DialoGPT-medium").to(self.device)
            except Exception:
                # Fallback to simple quantization
                self.compression_type = CompressionType.SIMPLE_QUANTIZATION
                print("Warning: Falling back to simple quantization")
        
        print(f"Initialized {self.compression_type.value} encoder")

    def encode(self, actions: Union[str, List[str], np.ndarray, torch.Tensor]) -> np.ndarray:
        """Encode actions to compressed latent representation.
        
        Args:
            actions: Action data to encode (text, vectors, or tensors)
            
        Returns:
            Compressed latent code as numpy array
        """
        start_time = time.time()
        
        try:
            # Convert input to appropriate format
            input_tensor = self._prepare_input(actions)
            
            if self.compression_type == CompressionType.LEARNED_VQVAE:
                latent_code = self._encode_vqvae(input_tensor)
            elif self.compression_type == CompressionType.NEURAL_COMPRESSION:
                latent_code = self._encode_neural(input_tensor)
            elif self.compression_type == CompressionType.DICTIONARY_COMPRESSION:
                latent_code = self._encode_dictionary(actions)
            else:  # SIMPLE_QUANTIZATION
                latent_code = self._encode_quantized(input_tensor)
            
            # Update metrics
            encoding_time = (time.time() - start_time) * 1000
            self._update_encoding_metrics(input_tensor, latent_code, encoding_time)
            
            return latent_code
            
        except Exception as e:
            print(f"Encoding failed: {e}")
            # Return safe default
            return np.zeros(self.latent_dim, dtype=np.float32)

    def decode(self, latent_code: np.ndarray) -> Union[np.ndarray, str]:
        """Decode latent code back to action representation.
        
        Args:
            latent_code: Compressed latent code
            
        Returns:
            Decoded action data
        """
        start_time = time.time()
        
        try:
            # Check for emergency codes first
            emergency_action = self._check_emergency_codes(latent_code)
            if emergency_action:
                return emergency_action
            
            if self.compression_type == CompressionType.LEARNED_VQVAE:
                decoded = self._decode_vqvae(latent_code)
            elif self.compression_type == CompressionType.NEURAL_COMPRESSION:
                decoded = self._decode_neural(latent_code)
            elif self.compression_type == CompressionType.DICTIONARY_COMPRESSION:
                decoded = self._decode_dictionary(latent_code)
            else:  # SIMPLE_QUANTIZATION
                decoded = self._decode_quantized(latent_code)
            
            # Update metrics
            decoding_time = (time.time() - start_time) * 1000
            self.metrics.decoding_time_ms = decoding_time
            
            return decoded
            
        except Exception as e:
            print(f"Decoding failed: {e}")
            # Return safe default
            return np.zeros(self.input_dim, dtype=np.float32)

    def encode_emergency_stop(self) -> np.ndarray:
        """Encode emergency stop command."""
        return self.emergency_codes['stop'].copy()

    def encode_emergency_land(self) -> np.ndarray:
        """Encode emergency land command."""
        return self.emergency_codes['land'].copy()

    def train(
        self,
        training_data: List[Any],
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Train the encoding model on demonstration data.
        
        Args:
            training_data: List of training samples
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Training batch size
            
        Returns:
            Training metrics and losses
        """
        if self.compression_type not in [CompressionType.LEARNED_VQVAE, CompressionType.NEURAL_COMPRESSION]:
            print("Training not supported for this compression type")
            return {}
        
        print(f"Training {self.compression_type.value} model...")
        
        # Prepare training data
        train_tensors = []
        for sample in training_data:
            tensor = self._prepare_input(sample)
            train_tensors.append(tensor)
        
        if not train_tensors:
            print("No valid training data")
            return {}
        
        train_data = torch.stack(train_tensors)
        
        # Set up training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        total_loss = 0.0
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                optimizer.zero_grad()
                
                if self.compression_type == CompressionType.LEARNED_VQVAE:
                    x_recon, z_e, z_q, _ = self.model(batch)
                    
                    # VQ-VAE loss
                    recon_loss = F.mse_loss(x_recon, batch)
                    vq_loss = F.mse_loss(z_q, z_e.detach())
                    commit_loss = F.mse_loss(z_e, z_q.detach())
                    
                    loss = recon_loss + vq_loss + self.model.commitment_cost * commit_loss
                    
                else:  # Neural compression
                    encoded = self.model(batch)
                    # For autoencoder, we need a decoder part
                    # For simplicity, assume identity mapping for now
                    loss = F.mse_loss(encoded, batch[:, :self.latent_dim])
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
        
        self.model.eval()
        self.is_trained = True
        
        avg_loss = total_loss / epochs
        print(f"Training completed. Average loss: {avg_loss:.4f}")
        
        return {
            'final_loss': avg_loss,
            'epochs': epochs,
            'compression_ratio': self._estimate_compression_ratio(train_data)
        }

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics.
        
        Returns:
            Comprehensive compression metrics
        """
        return {
            'compression_type': self.compression_type.value,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'theoretical_compression_ratio': self.input_dim / self.latent_dim,
            'is_trained': self.is_trained,
            'current_metrics': {
                'compression_ratio': self.metrics.compression_ratio,
                'encoding_time_ms': self.metrics.encoding_time_ms,
                'decoding_time_ms': self.metrics.decoding_time_ms,
                'reconstruction_error': self.metrics.reconstruction_error,
                'bandwidth_saved_percent': self.metrics.bandwidth_saved_percent,
            },
            'encoding_history_size': len(self.encoding_history),
            'average_encoding_time_ms': self._get_average_encoding_time(),
        }

    def _prepare_input(self, actions: Any) -> torch.Tensor:
        """Convert input to tensor format."""
        if isinstance(actions, str):
            # Convert text to embedding (simplified)
            # In real implementation, use proper text encoding
            embedding = np.random.randn(self.input_dim).astype(np.float32)
            return torch.from_numpy(embedding).to(self.device)
        
        elif isinstance(actions, (list, tuple)):
            if isinstance(actions[0], str):
                # Multiple text actions
                embeddings = []
                for action in actions:
                    embedding = np.random.randn(self.input_dim).astype(np.float32)
                    embeddings.append(embedding)
                combined = np.mean(embeddings, axis=0)
                return torch.from_numpy(combined).to(self.device)
            else:
                # Numeric list
                arr = np.array(actions, dtype=np.float32)
                if arr.shape[0] != self.input_dim:
                    # Pad or truncate
                    if arr.shape[0] < self.input_dim:
                        arr = np.pad(arr, (0, self.input_dim - arr.shape[0]))
                    else:
                        arr = arr[:self.input_dim]
                return torch.from_numpy(arr).to(self.device)
        
        elif isinstance(actions, np.ndarray):
            arr = actions.astype(np.float32)
            if arr.shape[0] != self.input_dim:
                if arr.shape[0] < self.input_dim:
                    arr = np.pad(arr, (0, self.input_dim - arr.shape[0]))
                else:
                    arr = arr[:self.input_dim]
            return torch.from_numpy(arr).to(self.device)
        
        elif isinstance(actions, torch.Tensor):
            return actions.to(self.device)
        
        else:
            # Default: random embedding
            embedding = np.random.randn(self.input_dim).astype(np.float32)
            return torch.from_numpy(embedding).to(self.device)

    def _encode_vqvae(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Encode using VQ-VAE model."""
        with torch.no_grad():
            indices = self.model.encode(input_tensor.unsqueeze(0))
            return indices.cpu().numpy().flatten()

    def _decode_vqvae(self, latent_code: np.ndarray) -> np.ndarray:
        """Decode using VQ-VAE model."""
        with torch.no_grad():
            indices = torch.from_numpy(latent_code.astype(np.long)).to(self.device)
            decoded = self.model.decode(indices.unsqueeze(0))
            return decoded.cpu().numpy().flatten()

    def _encode_neural(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Encode using neural compression."""
        with torch.no_grad():
            encoded = self.model(input_tensor.unsqueeze(0))
            return encoded.cpu().numpy().flatten()

    def _decode_neural(self, latent_code: np.ndarray) -> np.ndarray:
        """Decode using neural compression (simplified)."""
        # For MVP, use simple upsampling
        # In real implementation, would need proper decoder
        upsampled = np.repeat(latent_code, self.input_dim // len(latent_code))
        if len(upsampled) < self.input_dim:
            upsampled = np.pad(upsampled, (0, self.input_dim - len(upsampled)))
        return upsampled[:self.input_dim]

    def _encode_dictionary(self, actions: str) -> np.ndarray:
        """Encode using dictionary compression."""
        if self.tokenizer and self.model:
            try:
                tokens = self.tokenizer.encode(actions, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(tokens)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    
                # Compress to target dimension
                compressed = embedding.cpu().numpy().flatten()[:self.latent_dim]
                if len(compressed) < self.latent_dim:
                    compressed = np.pad(compressed, (0, self.latent_dim - len(compressed)))
                
                return compressed
            except Exception:
                pass
        
        # Fallback to simple quantization
        return self._encode_quantized(self._prepare_input(actions))

    def _decode_dictionary(self, latent_code: np.ndarray) -> str:
        """Decode using dictionary compression."""
        # Simplified decoding to action description
        # In real implementation, would use proper language model decoding
        action_templates = [
            "move forward",
            "turn left", 
            "turn right",
            "ascend",
            "descend",
            "hover",
            "land",
            "take off"
        ]
        
        # Use latent code to select action
        idx = int(np.sum(latent_code) * len(action_templates)) % len(action_templates)
        return action_templates[idx]

    def _encode_quantized(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Simple quantization encoding."""
        # Quantize to 8-bit and downsample
        arr = input_tensor.cpu().numpy()
        quantized = np.round(arr * 127).astype(np.int8)
        
        # Downsample to target dimension
        step = len(quantized) // self.latent_dim
        if step > 0:
            downsampled = quantized[::step][:self.latent_dim]
        else:
            downsampled = quantized[:self.latent_dim]
        
        # Pad if necessary
        if len(downsampled) < self.latent_dim:
            downsampled = np.pad(downsampled, (0, self.latent_dim - len(downsampled)))
        
        return downsampled.astype(np.float32)

    def _decode_quantized(self, latent_code: np.ndarray) -> np.ndarray:
        """Simple quantization decoding."""
        # Upsample and dequantize
        upsampled = np.repeat(latent_code, self.input_dim // len(latent_code))
        if len(upsampled) < self.input_dim:
            upsampled = np.pad(upsampled, (0, self.input_dim - len(upsampled)))
        
        dequantized = upsampled[:self.input_dim] / 127.0
        return dequantized.astype(np.float32)

    def _check_emergency_codes(self, latent_code: np.ndarray) -> Optional[str]:
        """Check if latent code matches emergency commands."""
        for command, code in self.emergency_codes.items():
            if len(latent_code) >= len(code):
                if np.allclose(latent_code[:len(code)], code, atol=1.0):
                    return f"EMERGENCY_{command.upper()}"
        return None

    def _update_encoding_metrics(
        self,
        input_tensor: torch.Tensor,
        latent_code: np.ndarray,
        encoding_time: float
    ) -> None:
        """Update encoding performance metrics."""
        input_size = input_tensor.numel() * input_tensor.element_size()
        latent_size = latent_code.nbytes
        
        compression_ratio = input_size / latent_size if latent_size > 0 else 0
        bandwidth_saved = ((input_size - latent_size) / input_size * 100) if input_size > 0 else 0
        
        self.metrics.compression_ratio = compression_ratio
        self.metrics.encoding_time_ms = encoding_time
        self.metrics.bandwidth_saved_percent = bandwidth_saved
        
        # Store in history
        self.encoding_history.append({
            'timestamp': time.time(),
            'compression_ratio': compression_ratio,
            'encoding_time_ms': encoding_time,
            'input_size_bytes': input_size,
            'latent_size_bytes': latent_size,
        })
        
        # Keep only last 100 entries
        if len(self.encoding_history) > 100:
            self.encoding_history = self.encoding_history[-100:]

    def _estimate_compression_ratio(self, data: torch.Tensor) -> float:
        """Estimate compression ratio from training data."""
        input_size = data.numel() * data.element_size()
        # Estimate latent size based on dimension
        latent_size = len(data) * self.latent_dim * 4  # float32
        return input_size / latent_size if latent_size > 0 else 0

    def _get_average_encoding_time(self) -> float:
        """Get average encoding time from history."""
        if not self.encoding_history:
            return 0.0
        
        times = [entry['encoding_time_ms'] for entry in self.encoding_history]
        return sum(times) / len(times)