"""
Tests for Phase 3 multimodal modules:
    - CrossModalAttention
    - MultimodalEncoder
    - AudioEncoder
"""

import pytest
import torch

from ucga.encoders import CrossModalAttention, MultimodalEncoder, AudioEncoder


BATCH = 4
DIM = 64


# ===========================================================================
# CrossModalAttention
# ===========================================================================
class TestCrossModalAttention:
    def test_forward_shape_2d(self):
        """2D inputs (B, D) should produce 2D outputs."""
        fuse = CrossModalAttention(dim=DIM, num_heads=4)
        a = torch.randn(BATCH, DIM)
        b = torch.randn(BATCH, DIM)
        out_a, out_b = fuse(a, b)
        assert out_a.shape == (BATCH, DIM)
        assert out_b.shape == (BATCH, DIM)

    def test_forward_shape_3d(self):
        """3D inputs (B, L, D) should produce 3D outputs."""
        fuse = CrossModalAttention(dim=DIM, num_heads=4)
        a = torch.randn(BATCH, 5, DIM)
        b = torch.randn(BATCH, 8, DIM)
        out_a, out_b = fuse(a, b)
        assert out_a.shape == (BATCH, 5, DIM)
        assert out_b.shape == (BATCH, 8, DIM)

    def test_gradient_flow(self):
        fuse = CrossModalAttention(dim=DIM, num_heads=4)
        a = torch.randn(BATCH, DIM, requires_grad=True)
        b = torch.randn(BATCH, DIM, requires_grad=True)
        out_a, out_b = fuse(a, b)
        (out_a.sum() + out_b.sum()).backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_bidirectional(self):
        """Both modalities should be modified by cross-attention."""
        fuse = CrossModalAttention(dim=DIM, num_heads=4)
        a = torch.randn(BATCH, DIM)
        b = torch.randn(BATCH, DIM)
        out_a, out_b = fuse(a, b)
        # Outputs should differ from inputs (cross-attention modifies them)
        assert not torch.allclose(out_a, a, atol=1e-3)
        assert not torch.allclose(out_b, b, atol=1e-3)


# ===========================================================================
# MultimodalEncoder
# ===========================================================================
class TestMultimodalEncoder:
    def test_forward_shape(self):
        enc = MultimodalEncoder(
            image_channels=3,
            vocab_size=200,
            embed_dim=DIM,
            output_dim=128,
            num_heads=4,
            num_fusion_layers=1,
            text_encoder_type="transformer",
            max_seq_len=16,
        )
        images = torch.randn(BATCH, 3, 32, 32)
        tokens = torch.randint(1, 200, (BATCH, 16))
        out = enc(images, tokens)
        assert out.shape == (BATCH, 128)

    def test_cnn_text_encoder(self):
        """Should also work with the CNN text encoder."""
        enc = MultimodalEncoder(
            image_channels=3,
            vocab_size=200,
            embed_dim=DIM,
            output_dim=128,
            num_heads=4,
            num_fusion_layers=1,
            text_encoder_type="cnn",
            max_seq_len=16,
        )
        images = torch.randn(BATCH, 3, 32, 32)
        tokens = torch.randint(1, 200, (BATCH, 16))
        out = enc(images, tokens)
        assert out.shape == (BATCH, 128)

    def test_gradient_flow(self):
        enc = MultimodalEncoder(
            image_channels=3,
            vocab_size=200,
            embed_dim=DIM,
            output_dim=128,
            num_heads=4,
            num_fusion_layers=1,
            text_encoder_type="transformer",
            max_seq_len=16,
        )
        images = torch.randn(BATCH, 3, 32, 32, requires_grad=True)
        tokens = torch.randint(1, 200, (BATCH, 16))
        out = enc(images, tokens)
        out.sum().backward()
        assert images.grad is not None

    def test_no_nan(self):
        enc = MultimodalEncoder(
            image_channels=1,
            vocab_size=100,
            embed_dim=DIM,
            output_dim=64,
            num_heads=4,
            num_fusion_layers=2,
            text_encoder_type="transformer",
            max_seq_len=8,
        )
        images = torch.randn(BATCH, 1, 28, 28)
        tokens = torch.randint(0, 100, (BATCH, 8))
        out = enc(images, tokens)
        assert not torch.isnan(out).any()


# ===========================================================================
# AudioEncoder
# ===========================================================================
class TestAudioEncoder:
    def test_forward_shape(self):
        enc = AudioEncoder(n_mels=128, output_dim=64, in_channels=1)
        mel = torch.randn(BATCH, 1, 128, 100)  # 100 time frames
        out = enc(mel)
        assert out.shape == (BATCH, 64)

    def test_auto_channel_dim(self):
        """3D input (B, n_mels, T) should auto-add channel dim."""
        enc = AudioEncoder(n_mels=64, output_dim=128)
        mel = torch.randn(BATCH, 64, 50)
        out = enc(mel)
        assert out.shape == (BATCH, 128)

    def test_variable_length(self):
        """Should handle different time lengths via adaptive pooling."""
        enc = AudioEncoder(n_mels=128, output_dim=64)
        for T in [30, 100, 500]:
            mel = torch.randn(BATCH, 1, 128, T)
            out = enc(mel)
            assert out.shape == (BATCH, 64)

    def test_stereo(self):
        """2-channel (stereo) input should work."""
        enc = AudioEncoder(n_mels=128, output_dim=64, in_channels=2)
        mel = torch.randn(BATCH, 2, 128, 100)
        out = enc(mel)
        assert out.shape == (BATCH, 64)

    def test_gradient_flow(self):
        enc = AudioEncoder(n_mels=64, output_dim=128)
        mel = torch.randn(BATCH, 1, 64, 80, requires_grad=True)
        out = enc(mel)
        out.sum().backward()
        assert mel.grad is not None
