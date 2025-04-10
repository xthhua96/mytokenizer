from collections import OrderedDict
from collections import defaultdict
import pickle
import re
from tqdm import tqdm


# Byte-Pair Encoding tokenization
class OptimizedBPETokenizer:
    def __init__(self):
        self.b2i = {}  # 普通字典替代 OrderedDict
        self.i2b = {}  # 反向映射
        self.b2i_set = set()  # 快速存在性检查
        self.next_id = 0

        # 特殊token相关
        self.sp_s2i = {}
        self.sp_i2s = {}
        self.split_re = None  # 预编译正则表达式
        self.sp_pattern = None

    # 预处理相邻对统计（使用 defaultdict）
    def _pair_stats(self, tokens, stats):
        for i in range(len(tokens) - 1):
            stats[tokens[i] + tokens[i + 1]] += 1

    # 优化合并逻辑（预计算长度）
    def _merge_pair(self, tokens, new_token):
        merged = []
        i, n = 0, len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] + tokens[i + 1] == new_token:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text_list, vocab_size):
        # 初始化单字节词表（字典推导式优化）
        self.b2i = {bytes([i]): i for i in range(256)}
        self.b2i_set = set(self.b2i.keys())
        self.next_id = 256

        # 预处理语料（生成器优化内存）
        tokens_list = [[bytes([b]) for b in text.encode("utf-8")] for text in text_list]

        with tqdm(total=vocab_size - 256) as progress:
            while self.next_id < vocab_size:
                # 使用 defaultdict 提升统计效率
                stats = defaultdict(int)
                for tokens in tokens_list:
                    self._pair_stats(tokens, stats)

                if not stats:
                    break

                # 直接使用 max 的 key 参数
                new_token = max(stats, key=stats.get)

                # 并行化合并操作
                tokens_list = [self._merge_pair(ts, new_token) for ts in tokens_list]

                # 批量更新字典
                self.b2i[new_token] = self.next_id
                self.b2i_set.add(new_token)
                self.next_id += 1
                progress.update(1)

        self.i2b = {v: k for k, v in self.b2i.items()}

    def add_special_tokens(self, special_tokens):
        # 动态更新正则表达式
        new_tokens = [tok for tok in special_tokens if tok not in self.sp_s2i]
        if new_tokens:
            for tok in new_tokens:
                self.sp_s2i[tok] = self.next_id
                self.sp_i2s[self.next_id] = tok
                self.next_id += 1
            # 更新预编译正则
            escaped = [re.escape(t) for t in self.sp_s2i]
            self.split_re = re.compile(f"({'|'.join(escaped)})")

    def encode(self, text):
        if self.split_re:
            splits = self.split_re.split(text)
        else:
            splits = [text]

        enc_ids = []
        enc_tokens = []
        for sub in splits:
            if sub in self.sp_s2i:
                enc_ids.append(self.sp_s2i[sub])
                enc_tokens.append(sub.encode())
                continue

            # 优化字节转换
            tokens = [bytes([b]) for b in sub.encode("utf-8")]
            merged = True
            while merged:
                # 使用生成器表达式优化内存
                stats = defaultdict(int)
                self._pair_stats(tokens, stats)

                # 使用 min + 生成器 优化选择逻辑
                valid_merges = (mt for mt in stats if mt in self.b2i_set)
                if new_token := min(
                    valid_merges, key=lambda x: self.b2i[x], default=None
                ):
                    tokens = self._merge_pair(tokens, new_token)
                else:
                    merged = False

            enc_ids.extend(self.b2i[t] for t in tokens)
            enc_tokens.extend(tokens)

        return enc_ids, enc_tokens

    # 词表大小
    def vocab_size(self):
        return self.next_id

    # 词表
    def vocab(self):
        v = {}
        v.update(self.i2b)
        v.update({id: token.encode("utf-8") for id, token in self.sp_i2s.items()})
        return v

    def decode(self, ids):
        bytes_list = []
        for id in ids:
            if id in self.sp_i2s:
                bytes_list.append(self.sp_i2s[id].encode("utf-8"))
            else:
                bytes_list.append(self.i2b[id])
        return b"".join(bytes_list).decode("utf-8", errors="replace")

    def save(self, file):
        with open(file, "wb") as fp:
            fp.write(pickle.dumps((self.b2i, self.sp_s2i, self.next_id)))

    def load(self, file):
        with open(file, "rb") as fp:
            self.b2i, self.sp_s2i, self.next_id = pickle.loads(fp.read())
        self.i2b = {v: k for k, v in self.b2i.items()}
        self.sp_i2s = {v: k for k, v in self.sp_s2i.items()}


if __name__ == "__main__":
    # 加载语料
    cn = open("dataset/train-cn.txt", "r").read()
    en = open("dataset/train-en.txt", "r").read()

    # 训练
    tokenizer = OptimizedBPETokenizer()
    tokenizer.train(text_list=[cn, en], vocab_size=5000)

    special_tokens = ["<SOS>", "<EOS>", "<CLS>", "<SEP>", "<PAD>", "<UNK>"]

    # 特殊token
    tokenizer.add_special_tokens(special_tokens)

    # 保存
    tokenizer.save("tokenizer.bin")

    # 还原
    tokenizer = OptimizedBPETokenizer()
    tokenizer.load("tokenizer.bin")
    print("vocab size:", tokenizer.vocab_size())

    # 编码
    ids, tokens = tokenizer.encode(
        "<|im_start|>system\nyou are a helper assistant\n<|im_end|>\n<|im_start|>user\n今天的天气\n<|im_end|><|im_start|>assistant\n"
    )
    print("encode:", ids, tokens)

    # 解码
    s = tokenizer.decode(ids)
    print("decode:", s)

    # 打印词典
    print("vocab:", tokenizer.vocab())
