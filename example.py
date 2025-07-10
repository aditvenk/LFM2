import torch
from lfm2.main import create_lfm2_model

def forward_pass_example():
    """Example demonstrating different forward pass scenarios with the LFM2 model.
    
    This example shows:
    1. Basic forward pass
    2. Forward pass with attention masks
    3. Forward pass with caching
    4. Forward pass with all outputs
    5. Forward pass with custom position IDs
    """

    
    # Create a small model for demonstration
    model = create_lfm2_model("350M", verbose=True)
    model.eval()  # Set to evaluation mode
    
    # Example parameters
    batch_size = 2
    seq_length = 32
    vocab_size = model.config.vocab_size
    
    # Create example inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print("\n1. Basic Forward Pass")
    print("-" * 50)
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Expected shape: [batch_size, seq_length, vocab_size]")
        print(f"Actual: {list(outputs['logits'].shape)}")
    
    print("\n2. Forward Pass with Attention Mask")
    print("-" * 50)
    # Create attention mask (example: mask out some tokens)
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[:, -4:] = 0  # Mask out last 4 tokens
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print(f"Output with masked attention shape: {outputs['logits'].shape}")
    
    print("\n3. Forward Pass with Caching")
    print("-" * 50)
    # Enable caching for faster generation
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True
        )
        print("Cache sizes for each layer:")
        for idx, past_kv in enumerate(outputs['past_key_values']):
            print(f"Layer {idx}: Key shape: {past_kv[0].shape}, Value shape: {past_kv[1].shape}")
    
    print("\n4. Forward Pass with All Outputs")
    print("-" * 50)
    # Get all hidden states and attention patterns
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=True
        )
        print("Available outputs:", outputs.keys())
        print(f"Number of hidden states: {len(outputs['hidden_states'])}")
        print(f"Number of attention patterns: {len(outputs['attentions'])}")
    
    print("\n5. Forward Pass with Custom Position IDs")
    print("-" * 50)
    # Create custom position IDs (example: reversed positions)
    position_ids = torch.arange(seq_length - 1, -1, -1)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids
        )
        print(f"Output with custom positions shape: {outputs['logits'].shape}")

if __name__ == "__main__":
    # Run the forward pass example
    forward_pass_example()
