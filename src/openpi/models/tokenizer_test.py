import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


@pytest.fixture(scope="module")
def fast_rtc_tokenizer():
    return _tokenizer.FASTTokenizer(max_len=512)


@pytest.mark.parametrize("delay", [0, 1, 4, 7])
def test_fast_rtc_roundtrip_and_prefix_loss_mask(fast_rtc_tokenizer, delay):
    tokenizer = fast_rtc_tokenizer
    rng = np.random.default_rng(42)
    actions = rng.uniform(-0.5, 0.5, (8, 7)).astype(np.float32)
    state = np.zeros(7, np.float32)
    prefix, suffix = actions[:delay], actions[delay:]
    tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_rtc(
        "pick book", state, prefix, suffix, delay=delay, suffix_horizon=len(suffix)
    )
    context, context_mask, _, inference_loss = tokenizer.tokenize_rtc(
        "pick book", state, prefix, None, delay=delay, suffix_horizon=len(suffix)
    )
    np.testing.assert_array_equal(tokens[mask & ~loss_mask], context[context_mask])
    np.testing.assert_array_equal(ar_mask.astype(bool), loss_mask)
    assert not inference_loss.any()
    assert not loss_mask[~mask].any()
    decoded = tokenizer.extract_rtc_actions(tokens[loss_mask], len(suffix), 7)
    assert decoded.shape == suffix.shape
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, suffix, atol=0.13)

    changed, _, _, changed_loss = tokenizer.tokenize_rtc(
        "pick book", state, -prefix, suffix, delay=delay, suffix_horizon=len(suffix)
    )
    # A different committed prefix must not re-encode the supervised future.
    np.testing.assert_array_equal(changed[changed_loss], tokens[loss_mask])


def test_fast_rtc_rejects_truncated_wrong_horizon_and_missing_marker(fast_rtc_tokenizer):
    tokenizer = fast_rtc_tokenizer
    actions = np.full((4, 7), 0.25, np.float32)
    tokens, _, _, targets = tokenizer.tokenize_rtc(
        "pick book", np.zeros(7), actions[:1], actions[1:], delay=1, suffix_horizon=3
    )
    generated = tokens[targets]
    with pytest.raises(ValueError, match="incomplete"):
        tokenizer.extract_rtc_actions(generated[:-1], 3, 7)
    with pytest.raises(ValueError, match="missing the Action marker"):
        tokenizer.extract_rtc_actions(np.zeros(8, np.int32), 3, 7)
    with pytest.raises(ValueError, match="suffix horizon"):
        tokenizer.extract_rtc_actions(generated, 4, 7)


def test_fast_rtc_overflow_fails_instead_of_truncating(fast_rtc_tokenizer, monkeypatch):
    monkeypatch.setattr(fast_rtc_tokenizer, "_max_len", 4)
    with pytest.raises(ValueError, match="max_token_len"):
        fast_rtc_tokenizer.tokenize_rtc(
            "pick book", np.zeros(7), np.zeros((1, 7)), np.ones((3, 7)), delay=1, suffix_horizon=3
        )
