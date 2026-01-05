#!/usr/bin/env python
"""
Simple unit test to verify the block-wise module replacement logic works.
This test creates mock objects to test the functionality without needing actual models.
"""
import sys
import torch
import torch.nn as nn

# Add the auto_round module to the path
sys.path.insert(0, '/home/runner/work/auto-round-fork/auto-round-fork')

from auto_round.modelling.replace_modules import ReplacementModuleBase, apply_replacements_to_block

# Create a mock module to be replaced
class MockOriginalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))
    
    def forward(self, x):
        return x @ self.weight

# Create a replacement module
class MockReplacementModule(ReplacementModuleBase):
    def __init__(self, original, config=None):
        super().__init__()
        self.original_weight = original.weight
        self.replaced = True
    
    def forward(self, x):
        return x @ self.original_weight
    
    @classmethod
    def original_module_class(cls):
        return "MockOriginalModule"
    
    @classmethod
    def from_original(cls, original, config=None):
        return cls(original, config)

# Create a simple model with blocks
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {})()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                MockOriginalModule(),
                nn.ReLU(),
                MockOriginalModule()
            )
            for _ in range(3)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def test_block_replacement():
    """Test that block-wise replacement works correctly."""
    print("=" * 80)
    print("Testing block-wise module replacement logic")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating mock model with 3 blocks...")
    model = MockModel()
    
    # Count original modules
    original_count = sum(1 for _, m in model.named_modules() if isinstance(m, MockOriginalModule))
    print(f"   Found {original_count} MockOriginalModule instances")
    
    # Replace modules in first block only
    block_name = "blocks.0"
    print(f"\n2. Applying replacement to block: {block_name}")
    replaced_count = apply_replacements_to_block(model, block_name)
    print(f"   Replaced {replaced_count} modules in '{block_name}'")
    
    # Count modules after first replacement
    block0 = model.get_submodule(block_name)
    replaced_in_block0 = sum(1 for _, m in block0.named_modules() if isinstance(m, MockReplacementModule))
    original_in_block0 = sum(1 for _, m in block0.named_modules() if isinstance(m, MockOriginalModule))
    print(f"   Block 0: {replaced_in_block0} replaced, {original_in_block0} original")
    
    # Check other blocks are unchanged
    block1 = model.get_submodule("blocks.1")
    original_in_block1 = sum(1 for _, m in block1.named_modules() if isinstance(m, MockOriginalModule))
    print(f"   Block 1: {original_in_block1} original (should be 2)")
    
    # Replace modules in second block
    block_name_1 = "blocks.1"
    print(f"\n3. Applying replacement to block: {block_name_1}")
    replaced_count_1 = apply_replacements_to_block(model, block_name_1)
    print(f"   Replaced {replaced_count_1} modules in '{block_name_1}'")
    
    # Count modules after second replacement
    replaced_in_block1 = sum(1 for _, m in block1.named_modules() if isinstance(m, MockReplacementModule))
    original_in_block1 = sum(1 for _, m in block1.named_modules() if isinstance(m, MockOriginalModule))
    print(f"   Block 1: {replaced_in_block1} replaced, {original_in_block1} original")
    
    # Final verification
    total_replaced = sum(1 for _, m in model.named_modules() if isinstance(m, MockReplacementModule))
    total_original = sum(1 for _, m in model.named_modules() if isinstance(m, MockOriginalModule))
    
    print(f"\n4. Final counts:")
    print(f"   Total replaced: {total_replaced}")
    print(f"   Total original: {total_original}")
    
    # Verify results
    print(f"\n5. Verification:")
    success = True
    
    if replaced_count == 2:
        print("   ✓ Block 0 replacement count correct (2)")
    else:
        print(f"   ✗ Block 0 replacement count incorrect (expected 2, got {replaced_count})")
        success = False
    
    if replaced_count_1 == 2:
        print("   ✓ Block 1 replacement count correct (2)")
    else:
        print(f"   ✗ Block 1 replacement count incorrect (expected 2, got {replaced_count_1})")
        success = False
    
    if original_in_block0 == 0:
        print("   ✓ No original modules remain in block 0")
    else:
        print(f"   ✗ Original modules still present in block 0 ({original_in_block0})")
        success = False
    
    if original_in_block1 == 0:
        print("   ✓ No original modules remain in block 1")
    else:
        print(f"   ✗ Original modules still present in block 1 ({original_in_block1})")
        success = False
    
    # Block 2 should still have original modules (2)
    block2 = model.get_submodule("blocks.2")
    original_in_block2 = sum(1 for _, m in block2.named_modules() if isinstance(m, MockOriginalModule))
    if original_in_block2 == 2:
        print("   ✓ Block 2 still has original modules (not replaced)")
    else:
        print(f"   ✗ Block 2 state incorrect (expected 2 original, got {original_in_block2})")
        success = False
    
    if success:
        print("\n" + "=" * 80)
        print("✓ All tests passed! Block-wise replacement logic works correctly.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ Some tests failed!")
        print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_block_replacement()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
