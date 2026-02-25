"""
test_encoders.py — Unit tests for all UCGA input encoders.
"""

import pytest
import torch

from ucga.encoders import TextEncoder, ImageEncoder, VectorEncoder, TransformerTextEncoder


BATCH = 4


class TestVectorEncoder:
    def test_forward_shape(self):
        enc = VectorEncoder(input_dim=32, output_dim=64)
        x = torch.randn(BATCH, 32)
        out = enc(x)
        assert out.shape == (BATCH, 64)

    def test_different_dims(self):
        enc = VectorEncoder(input_dim=100, output_dim=256, hidden_dim=128)
        x = torch.randn(BATCH, 100)
        out = enc(x)
        assert out.shape == (BATCH, 256)

    def test_gradient_flow(self):
        enc = VectorEncoder(input_dim=32, output_dim=64)
        x = torch.randn(BATCH, 32, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestTextEncoder:
    def test_forward_shape(self):
        enc = TextEncoder(vocab_size=100, embed_dim=32, output_dim=64, max_seq_len=16)
        ids = torch.randint(0, 100, (BATCH, 16))
        out = enc(ids)
        assert out.shape == (BATCH, 64)

    def test_padding_handling(self):
        """Sequences with padding (zeros) should produce valid output."""
        enc = TextEncoder(vocab_size=100, embed_dim=32, output_dim=64, max_seq_len=16)
        ids = torch.zeros(BATCH, 16, dtype=torch.long)
        ids[:, :3] = torch.randint(1, 100, (BATCH, 3))
        out = enc(ids)
        assert out.shape == (BATCH, 64)
        assert not torch.isnan(out).any()

    def test_single_token(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, output_dim=32, max_seq_len=1)
        ids = torch.randint(0, 50, (BATCH, 1))
        out = enc(ids)
        assert out.shape == (BATCH, 32)


class TestTransformerTextEncoder:
    def test_forward_shape(self):
        enc = TransformerTextEncoder(
            vocab_size=200, embed_dim=32, output_dim=64,
            max_seq_len=16, num_layers=1, num_heads=2,
        )
        ids = torch.randint(1, 200, (BATCH, 16))
        out = enc(ids)
        assert out.shape == (BATCH, 64)

    def test_padding_handling(self):
        """Padded sequences (index 0) should produce valid output."""
        enc = TransformerTextEncoder(
            vocab_size=200, embed_dim=32, output_dim=64,
            max_seq_len=16, num_layers=1, num_heads=2,
        )
        ids = torch.zeros(BATCH, 16, dtype=torch.long)
        ids[:, :5] = torch.randint(1, 200, (BATCH, 5))
        out = enc(ids)
        assert out.shape == (BATCH, 64)
        assert not torch.isnan(out).any()

    def test_variable_length(self):
        """Different sequence lengths should all work."""
        for seq_len in [1, 4, 16]:
            enc = TransformerTextEncoder(
                vocab_size=100, embed_dim=32, output_dim=48,
                max_seq_len=16, num_layers=1, num_heads=2,
            )
            ids = torch.randint(1, 100, (BATCH, seq_len))
            out = enc(ids)
            assert out.shape == (BATCH, 48)

    def test_gradient_flow(self):
        """Gradients should flow through the encoder."""
        enc = TransformerTextEncoder(
            vocab_size=100, embed_dim=32, output_dim=64,
            max_seq_len=16, num_layers=1, num_heads=2,
        )
        ids = torch.randint(1, 100, (BATCH, 8))
        out = enc(ids)
        loss = out.sum()
        loss.backward()
        # Check that embedding weights got gradients
        assert enc.token_embedding.weight.grad is not None

    def test_explicit_mask(self):
        """Providing an explicit attention mask should work."""
        enc = TransformerTextEncoder(
            vocab_size=100, embed_dim=32, output_dim=64,
            max_seq_len=16, num_layers=1, num_heads=2,
        )
        ids = torch.randint(1, 100, (BATCH, 8))
        mask = torch.zeros(BATCH, 8, dtype=torch.bool)
        mask[:, 5:] = True  # mask last 3 tokens
        out = enc(ids, attention_mask=mask)
        assert out.shape == (BATCH, 64)
        assert not torch.isnan(out).any()


class TestImageEncoder:
    def test_forward_shape(self):
        enc = ImageEncoder(in_channels=3, output_dim=64)
        img = torch.randn(BATCH, 3, 32, 32)
        out = enc(img)
        assert out.shape == (BATCH, 64)

    def test_grayscale(self):
        enc = ImageEncoder(in_channels=1, output_dim=128)
        img = torch.randn(BATCH, 1, 28, 28)
        out = enc(img)
        assert out.shape == (BATCH, 128)

    def test_gradient_flow(self):
        enc = ImageEncoder(in_channels=3, output_dim=64)
        img = torch.randn(BATCH, 3, 32, 32, requires_grad=True)
        out = enc(img)
        out.sum().backward()
        assert img.grad is not None
