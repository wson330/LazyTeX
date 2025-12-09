from typing import Dict, List


class CROHMEVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def init(self, dict_path: str) -> None:
        self.word2idx: Dict[str, int] = {
            "<pad>": self.PAD_IDX,
            "<sos>": self.SOS_IDX,
            "<eos>": self.EOS_IDX,
        }
        with open(dict_path, "r", encoding="utf-8") as handle:
            for line in handle:
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        return " ".join(self.indices2words(id_list))

    def __len__(self) -> int:
        return len(self.word2idx)


vocab = CROHMEVocab()
