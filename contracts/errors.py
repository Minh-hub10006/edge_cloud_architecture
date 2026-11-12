"""Preserve Cloud rejection status independently of transport failures."""


class CloudInferenceRejected(RuntimeError):
    def __init__(self, status: str, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"Cloud inference {status}: {detail}")
