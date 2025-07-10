import torch
from lfm2.main import create_lfm2_model, LFM2Config, LFM2Model, LFM2ForCausalLM

def test_model_creation():
    """Test basic model creation"""
    model = create_lfm2_model("350M", verbose=False)
    assert isinstance(model, LFM2ForCausalLM)
    assert model.config.hidden_size == 768
    print("✓ Model creation test passed")

def test_forward_pass():
    """Test basic forward pass"""
    model = create_lfm2_model("350M", verbose=False)
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, model.config.vocab_size)
    print("✓ Forward pass test passed")

def test_attention_mask():
    """Test forward pass with attention mask"""
    model = create_lfm2_model("350M", verbose=False)
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # Mask out second half
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    assert "logits" in outputs
    print("✓ Attention mask test passed")

def test_config():
    """Test model configuration"""
    config = LFM2Config.from_pretrained_size("700M")
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 16
    assert config.num_conv_blocks == 10
    assert config.num_attention_blocks == 6
    print("✓ Config test passed")

def run_all_tests():
    """Run all test functions"""
    print("\nRunning LFM2 tests...")
    test_model_creation()
    test_forward_pass()
    test_attention_mask()
    test_config()
    print("\nAll tests passed! ✨")

if __name__ == "__main__":
    run_all_tests() 