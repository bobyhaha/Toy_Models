class CharTokenizer:
    def __init__(self):
        chars = list("0123456789.-+=;xy ")
        special = ["<pad>", "<bos>", "<eos>"]
        self.itos = special + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s, add_bos=True, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in s:
            if ch not in self.stoi:
                raise ValueError(f"Unknown character {ch!r} in {s!r}")
            ids.append(self.stoi[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            ch = self.itos[int(i)]
            if ch in ["<pad>", "<bos>", "<eos>"]:
                continue
            out.append(ch)
        return "".join(out)
