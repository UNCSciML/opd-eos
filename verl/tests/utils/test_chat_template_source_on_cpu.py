import hashlib

from verl.utils import tokenizer as tokenizer_utils


class StudentTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    chat_template = "student {{ enable_thinking }}"

    def apply_chat_template(self, messages, **kwargs):
        assert self.eos_token_id == 1
        return (messages[0]["content"], kwargs)


class SourceTokenizer:
    eos_token_id = 999
    pad_token_id = 998
    chat_template = "source template without a thinking switch"


def test_external_chat_template_loads_text_only_and_preserves_student_special_tokens(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SourceTokenizer(),
    )
    student = StudentTokenizer()

    template, digest = tokenizer_utils.load_chat_template_text("/models/instruct")
    kwargs = tokenizer_utils.resolve_chat_template_kwargs(
        tokenizer=student,
        chat_template=template,
        enable_thinking=False,
    )
    rendered = student.apply_chat_template([{"role": "user", "content": "question"}], **kwargs)

    assert template == SourceTokenizer.chat_template
    assert digest == hashlib.sha256(template.encode("utf-8")).hexdigest()
    assert student.eos_token_id == 1
    assert student.pad_token_id == 0
    assert rendered[1]["chat_template"] == template
    assert "enable_thinking" not in rendered[1]


def test_eos_mode_does_not_enter_chat_template_rendering_kwargs():
    student = StudentTokenizer()
    rows = []
    for _mode in ("baseline", "two_stop", "semantic_class"):
        kwargs = tokenizer_utils.resolve_chat_template_kwargs(
            tokenizer=student,
            chat_template=student.chat_template,
            enable_thinking=False,
        )
        rows.append(student.apply_chat_template([{"role": "user", "content": "same"}], **kwargs))

    assert rows[0] == rows[1] == rows[2]
