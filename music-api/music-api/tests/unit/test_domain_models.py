from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.models import (
    Genre,
    Interaction,
    InteractionType,
    Track,
    TrackCreate,
    UserCreate,
    INTERACTION_WEIGHTS,
)


class TestUserCreate:
    def test_valid_user_passes(self):
        u = UserCreate(
            username="ada_lovelace",
            email="ada@test.com",
            display_name="Ada",
            password="Senha123",
            confirm_password="Senha123",
        )
        assert u.username == "ada_lovelace"

    def test_password_mismatch_raises(self):
        with pytest.raises(ValidationError, match="senhas não coincidem"):
            UserCreate(
                username="ada",
                email="ada@test.com",
                display_name="Ada",
                password="Senha123",
                confirm_password="Outra123",
            )

    def test_weak_password_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(
                username="ada",
                email="ada@test.com",
                display_name="Ada",
                password="somenumbers123",
                confirm_password="somenumbers123",
            )

    def test_invalid_username_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(
                username="user name!",
                email="test@test.com",
                display_name="Test",
                password="Senha123",
                confirm_password="Senha123",
            )

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(
                username="user",
                email="not-an-email",
                display_name="Test",
                password="Senha123",
                confirm_password="Senha123",
            )


class TestTrackCreate:
    def test_valid_track(self):
        t = TrackCreate(
            title="Song",
            artist_id=uuid4(),
            duration_ms=180_000,
            genres=[Genre.POP],
            danceability=0.8,
            energy=0.9,
            valence=0.7,
            acousticness=0.1,
            instrumentalness=0.0,
            tempo_bpm=128.0,
            loudness_db=-6.0,
        )
        assert t.title == "Song"

    def test_duration_too_short_raises(self):
        with pytest.raises(ValidationError):
            TrackCreate(
                title="Short",
                artist_id=uuid4(),
                duration_ms=500,
                genres=[Genre.POP],
            )

    def test_audio_feature_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            TrackCreate(
                title="Bad",
                artist_id=uuid4(),
                duration_ms=180_000,
                genres=[Genre.POP],
                danceability=1.5,
            )

    def test_feature_vector_has_correct_length(self):
        from tests.conftest import make_track_create
        t = make_track_create()
        track = Track(**t.model_dump(), id=uuid4())
        assert len(track.feature_vector) == 8


class TestInteractionWeight:
    def test_like_has_positive_weight(self):
        i = Interaction(
            user_id=uuid4(),
            track_id=uuid4(),
            interaction_type=InteractionType.LIKE,
        )
        assert i.weight > 0

    def test_skip_has_negative_weight(self):
        i = Interaction(
            user_id=uuid4(),
            track_id=uuid4(),
            interaction_type=InteractionType.SKIP,
        )
        assert i.weight < 0

    def test_play_with_full_duration_gives_full_weight(self):
        i = Interaction(
            user_id=uuid4(),
            track_id=uuid4(),
            interaction_type=InteractionType.PLAY,
            play_duration_ms=30_000,
        )
        assert i.weight == pytest.approx(INTERACTION_WEIGHTS[InteractionType.PLAY], abs=0.01)

    def test_play_with_partial_duration_scales_weight(self):
        i = Interaction(
            user_id=uuid4(),
            track_id=uuid4(),
            interaction_type=InteractionType.PLAY,
            play_duration_ms=15_000,
        )
        assert i.weight == pytest.approx(0.5, abs=0.01)

    def test_share_is_highest_positive_weight(self):
        share_w = INTERACTION_WEIGHTS[InteractionType.SHARE]
        play_w = INTERACTION_WEIGHTS[InteractionType.PLAY]
        like_w = INTERACTION_WEIGHTS[InteractionType.LIKE]
        assert share_w > like_w > play_w
