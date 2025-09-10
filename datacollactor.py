@dataclass
class OnlyAnswerLossCollator:
    tokenizer: Any
    response_template: str
    max_length: int
    padding: str = "longest"
    truncation: bool = True

    def __post_init__(self):
        ids = self.tokenizer(
            self.response_template, add_special_tokens=False, return_tensors=None
        )["input_ids"]
        # store as tuple to use with sliding window search
        self.template_ids = tuple(ids)

    def _find_subseq_end(self, seq: List[int], subseq: tuple) -> Optional[int]:
        """Return index after the end of the first occurrence of subseq in seq, else None."""
        n, m = len(seq), len(subseq)
        if m == 0 or m > n: 
            return None
        for i in range(n - m + 1):
            if tuple(seq[i:i+m]) == subseq:
                return i + m
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["instructions"] for ex in features]
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        # mask everything up to and including the response template
        for i, ids in enumerate(batch["input_ids"].tolist()):
            end = self._find_subseq_end(ids, self.template_ids)
            if end is None:
                # if the marker is missing, mask all (safer than training on prompt)
                labels[i, :] = -100
            else:
                labels[i, :end] = -100  # no loss on the prompt/template
        batch["labels"] = labels
        return batch