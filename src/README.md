"# Source Code" 
1. Special Characters
	<PAD> = 0
	<UNK> = 1
2. Determinism
	Vocab will be sorted by frequency, then character ascending.
3. Unknown Handling
	While strict is False, unknown characters will be replaced with <UNK>. If strict is True, an error will be raised for testing purposes.
4. Normalization
	Keep raw characters, no normalization.
5. Persistence
	Provide save and load methods for the vocab mapping.

To help define what this tokenizer will do, I will break down some simple notes.
A tokenizer is a tool that converts text into smaller units called tokens. These tokens, in this case, will be strictly characters.
These characters will be split based on whitespace and punctuation, then mapped directly to a numeric ID. 
For ease of implementation, I will use a dictionary to store the mapping of characters to IDs.
Padding will be represented by the special token <PAD> with an ID of 0, and unknown characters will be represented by <UNK> with an ID of 1.
Input characters will be counted, then sorted by frequency (most commonn first), then by character ascending to ensure determinism.
While strict is False, unknown characters will be replaced with <UNK>. If strict is True, an error will be raised for testing purposes.
The tokenizer will also provide methods to save and load the vocabulary mapping to and from a file for persistence.
The tokenizer will not perform any normalization on the characters, keeping them in their raw form.
