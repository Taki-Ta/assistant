import pytest

from interview_ai.indexing.searching import cosine_similarity


def test_cosine_similarity_of_same_direction_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    assert cosine_similarity([1.0, 0.0], [-2.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_different_dimensions():
    with pytest.raises(ValueError, match="向量长度不一致"):
        cosine_similarity([1.0], [1.0, 0.0])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
        ([], []),
    ],
)
def test_cosine_similarity_rejects_zero_vectors(a, b):
    with pytest.raises(ValueError, match="零向量"):
        cosine_similarity(a, b)
