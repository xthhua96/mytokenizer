import regex
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Union
import os
import re
import ast
from tqdm import tqdm


class SelfBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.special_tokens = {}
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        self.compiled_pattern = regex.compile(self.pattern, flags=regex.UNICODE)
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        # Add default special tokens
        self.add_special_tokens(
            [
                self.unk_token,
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.cls_token,
                self.sep_token,
                self.mask_token,
            ]
        )

    def add_special_tokens(self, tokens: List[str]) -> None:
        """Add special tokens to the tokenizer"""
        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = len(self.vocab) + len(self.special_tokens)
        escaped = [re.escape(token) for token in tokens]
        self.split_special_tokens_re = re.compile(f"({'|'.join(escaped)})")

    def train(
        self, texts: List[str], vocab_size: int = 30000, min_frequency: int = 2
    ) -> None:
        """Train the BPE tokenizer on given texts"""
        # Get initial vocabulary
        self.vocab = self._get_initial_vocab()

        byte_list = []
        for text in texts:
            for b in text.encode("utf-8"):
                byte_list.append(bytes([b]))

        max_iterations = vocab_size - len(self.vocab) - len(self.special_tokens)
        with tqdm(
            total=max_iterations, desc="Training SelfBPETokenizer", unit="merge"
        ) as pbar:
            # Perform BPE merges until we reach the desired vocab size
            while len(self.vocab) + len(self.special_tokens) < vocab_size:
                stats = defaultdict(int)
                self._get_pairs(byte_list, stats)
                # Get the most frequent pair
                max_pair = max(stats, key=stats.get)
                # Stop if frequency is below threshold
                if stats[max_pair] < min_frequency:
                    break
                # Merge the pair
                new_byte = self.tuple_to_bytes(max_pair)
                try:
                    self.merges.append(max_pair)
                    self.vocab[new_byte.decode("utf-8")] = len(self.vocab) + len(
                        self.special_tokens
                    )
                except UnicodeDecodeError:
                    pass
                # Update words and pairs
                byte_list = self._merge_pair(byte_list, max_pair, new_byte)
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "vocab_size": len(self.vocab) + len(self.special_tokens),
                        "top_pair": max_pair,
                        "pair_freq": stats.get(max_pair, 0),
                    }
                )

    def _get_initial_vocab(self) -> Dict[str, int]:
        vocab = {bytes([b]).decode("utf-8"): i for i, b in enumerate(range(128))}
        return vocab

    def _get_pairs(self, byte_list, stats) -> Dict[Tuple[str, str], int]:
        for i in range(len(byte_list) - 1):
            stats[byte_list[i], byte_list[i + 1]] += 1

    def _merge_pair(
        self, byte_list: List[bytes], pair: Tuple[bytes, bytes], new_byte: bytes
    ) -> List[bytes]:
        new_byte_list = []
        i = 0
        while i < len(byte_list) - 1:
            try:
                new_byte.decode("utf-8")
                if (
                    byte_list[i] == pair[0]
                    and byte_list[i + 1] == pair[1]
                    and new_byte.decode("utf-8") not in self.special_tokens
                ):
                    new_byte_list.append(new_byte)
                    i += 2
                else:
                    new_byte_list.append(byte_list[i])
                    i += 1
            except UnicodeDecodeError:
                if byte_list[i] == pair[0] and byte_list[i + 1] == pair[1]:
                    new_byte_list.append(new_byte)
                    i += 2
                else:
                    new_byte_list.append(byte_list[i])
                    i += 1
                continue

        return new_byte_list

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text using the trained BPE model"""
        if self.split_special_tokens_re:
            splits = self.split_special_tokens_re.split(text)
        else:
            splits = [text]

        result = []
        for split in splits:
            if split in self.special_tokens:
                result.append(split)
                continue

            byte_list = [bytes([b]) for b in split.encode("utf-8")]

            # Apply all merges
            for merge in self.merges:
                new_byte = []
                i = 0
                while i < len(byte_list):
                    if (
                        i < len(byte_list) - 1
                        and byte_list[i] == merge[0]
                        and byte_list[i + 1] == merge[1]
                    ):
                        new_byte.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_byte.append(byte_list[i])
                        i += 1
                byte_list = new_byte

            for byte_str in byte_list:
                try:
                    result.append(byte_str.decode("utf-8"))
                except UnicodeDecodeError:
                    result.append(self.unk_token)
        return result

    def encode(self, text: str) -> List[int]:
        """Convert text to token ids"""
        tokens = self.tokenize(text)
        return [self._convert_token_to_id(token) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token ids back to text"""
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        # Reconstruct text by joining tokens (simple approach, may need improvement)
        return "".join(tokens)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its id"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.vocab:
            return self.vocab[token]
        return self.special_tokens.get(self.unk_token, 0)

    def _convert_id_to_token(self, token_id: int) -> str:
        """Convert an id to its token"""
        # Check special tokens first
        for token, id_ in self.special_tokens.items():
            if id_ == token_id:
                return token
        # Then check vocab
        for token, id_ in self.vocab.items():
            if id_ == token_id:
                return token
        return self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        """Get the combined vocabulary (special tokens + learned tokens)"""
        return {**self.vocab, **self.special_tokens}

    def save(self, save_directory: str, name: Optional[str] = None) -> None:
        """Save the tokenizer files in the given directory"""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Determine file names
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.txt")
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        special_tokens_file = os.path.join(save_directory, "special_tokens_map.json")

        # Save vocab
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=2)

        # Save merges
        with open(merges_file, "w", encoding="utf-8") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")

        # Save config
        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "tokenizer_class": "SelfBPETokenizer",
            "model_max_length": 512,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Save special tokens map
        special_tokens_map = {
            "unk_token": {"content": self.unk_token, "lstrip": False, "rstrip": False},
            "pad_token": {"content": self.pad_token, "lstrip": False, "rstrip": False},
            "bos_token": {"content": self.bos_token, "lstrip": False, "rstrip": False},
            "eos_token": {"content": self.eos_token, "lstrip": False, "rstrip": False},
            "cls_token": {"content": self.cls_token, "lstrip": False, "rstrip": False},
            "sep_token": {"content": self.sep_token, "lstrip": False, "rstrip": False},
            "mask_token": {
                "content": self.mask_token,
                "lstrip": False,
                "rstrip": False,
            },
        }
        with open(special_tokens_file, "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "SelfBPETokenizer":
        """Load a pretrained tokenizer from the given directory"""
        tokenizer = cls()

        # Load vocab
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Separate special tokens and regular vocab
        tokenizer.vocab = {}
        tokenizer.special_tokens = {}
        for token, id_ in vocab.items():
            if token in {
                tokenizer.unk_token,
                tokenizer.pad_token,
                tokenizer.bos_token,
                tokenizer.eos_token,
                tokenizer.cls_token,
                tokenizer.sep_token,
                tokenizer.mask_token,
            }:
                tokenizer.special_tokens[token] = id_
            else:
                tokenizer.vocab[token] = id_

        # Load merges
        merges_file = os.path.join(save_directory, "merges.txt")
        with open(merges_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        tokenizer.merges.append(
                            (ast.literal_eval(parts[0]), ast.literal_eval(parts[1]))
                        )

        return tokenizer

    def tuple_to_bytes(self, data: tuple[Union[int, bytes]]) -> bytes:
        result = bytearray()
        for item in data:
            if isinstance(item, int):
                # 确保数字在 0~255 范围内
                if not 0 <= item <= 255:
                    raise ValueError(f"Integer {item} exceeds byte range (0-255)")
                result.append(item)
            elif isinstance(item, bytes):
                result.extend(item)
            else:
                raise TypeError(f"Unsupported type: {type(item)}")
        return bytes(result)


# 示例用法
if __name__ == "__main__":
    # 1. 创建并训练tokenizer
    tokenizer = SelfBPETokenizer()

    # # 示例训练数据
    # texts = [
    #     "This is an example sentence for training the BPE tokenizer.",
    #     "Another example with different words to increase the vocabulary.",
    #     "The quick brown fox jumps over the lazy dog.",
    #     "I love natural language processing and machine learning!",
    # ]
    # # 训练数据
    # with open("dataset/train-cn.txt", "r") as cn_file:
    #     cn = cn_file.read()
    # with open("dataset/train-en.txt", "r") as en_file:
    #     en = en_file.read()
    # texts = [cn, en]
    # # 训练tokenizer
    # tokenizer.train(texts=texts, vocab_size=10000, min_frequency=2)

    # # 2. 保存tokenizer
    # tokenizer.save("my_bpe_tokenizer")

    # 3. 从保存的文件加载tokeniz
    loaded_tokenizer = SelfBPETokenizer.from_pretrained("my_bpe_tokenizer")

    # 4. 测试tokenizer
    test_text = "T我爱你，你爱我吗？"

    print("Original text:", test_text)
    print("Tokenized:", loaded_tokenizer.tokenize(test_text))
    print("Encoded:", loaded_tokenizer.encode(test_text))
    print("Decoded:", loaded_tokenizer.decode(loaded_tokenizer.encode(test_text)))


# from transformers import RobertaTokenizer

# # 加载自定义 BPE Tokenizer
# tokenizer = RobertaTokenizer(
#     vocab_file="./my_bpe_tokenizer/vocab.json",
#     merges_file="./my_bpe_tokenizer/merges.txt",
# )

# # 测试分词
# print(tokenizer.tokenize("我爱你，你爱我吗？"))

# # 5. 检查与Hugging Face的兼容性
# from transformers import PreTrainedTokenizerFast

# 使用我们的文件创建Hugging Face tokenizer
# hf_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="my_bpe_tokenizer/vocab.json",
#     merges_file="my_bpe_tokenizer/merges.txt",
#     unk_token="<unk>",
#     pad_token="<pad>",
#     bos_token="<bos>",
#     eos_token="<eos>",
#     cls_token="<cls>",
#     sep_token="<sep>",
#     mask_token="<mask>",
# )

# print("\nHugging Face Tokenizer results:")
# print("Tokenized:", hf_tokenizer.tokenize(test_text))
# print("Encoded:", hf_tokenizer(test_text)["input_ids"])
# print("Decoded:", hf_tokenizer.decode(hf_tokenizer(test_text)["input_ids"]))
