from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator, model_validator
from typing_extensions import Self


# ── Enums ─────────────────────────────────────────────────────────────────────

class Genre(StrEnum):
    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    RNB = "rnb"
    COUNTRY = "country"
    METAL = "metal"
    FOLK = "folk"
    LATIN = "latin"
    REGGAE = "reggae"


class InteractionType(StrEnum):
    PLAY = "play"
    SKIP = "skip"
    LIKE = "like"
    DISLIKE = "dislike"
    SAVE = "save"
    SHARE = "share"
    ADD_TO_PLAYLIST = "add_to_playlist"


class NodeType(StrEnum):
    USER = "user"
    TRACK = "track"
    ARTIST = "artist"
    ALBUM = "album"
    GENRE = "genre"
    PLAYLIST = "playlist"


# ── Base ──────────────────────────────────────────────────────────────────────

PositiveFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class TimestampMixin(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Artist ────────────────────────────────────────────────────────────────────

class ArtistBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    genres: list[Genre] = Field(default_factory=list, max_length=10)
    popularity: PositiveFloat = 0.0
    country: str | None = Field(default=None, max_length=2)


class ArtistCreate(ArtistBase):
    pass


class Artist(ArtistBase, TimestampMixin):
    id: UUID = Field(default_factory=uuid4)
    follower_count: int = Field(default=0, ge=0)
    embedding: list[float] | None = None


# ── Track ─────────────────────────────────────────────────────────────────────

class TrackBase(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    artist_id: UUID
    album_id: UUID | None = None
    duration_ms: int = Field(ge=1000, le=3_600_000)
    genres: list[Genre] = Field(default_factory=list, max_length=5)

    # Audio features — normalised 0–1
    danceability: PositiveFloat = 0.5
    energy: PositiveFloat = 0.5
    valence: PositiveFloat = 0.5
    acousticness: PositiveFloat = 0.5
    instrumentalness: PositiveFloat = 0.5
    tempo_bpm: float = Field(default=120.0, ge=40.0, le=300.0)
    loudness_db: float = Field(default=-10.0, ge=-60.0, le=0.0)

    popularity: PositiveFloat = 0.0
    explicit: bool = False


class TrackCreate(TrackBase):
    pass


class Track(TrackBase, TimestampMixin):
    id: UUID = Field(default_factory=uuid4)
    play_count: int = Field(default=0, ge=0)
    embedding: list[float] | None = None

    @property
    def feature_vector(self) -> list[float]:
        return [
            self.danceability,
            self.energy,
            self.valence,
            self.acousticness,
            self.instrumentalness,
            self.tempo_bpm / 300.0,
            (self.loudness_db + 60.0) / 60.0,
            self.popularity,
        ]


# ── User ──────────────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_\-]+$")
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=100)


class UserCreate(UserBase):
    password: SecretStr = Field(min_length=8, max_length=128)
    confirm_password: SecretStr

    @model_validator(mode="after")
    def passwords_match(self) -> Self:
        if self.password.get_secret_value() != self.confirm_password.get_secret_value():
            raise ValueError("As senhas não coincidem")
        return self

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: SecretStr) -> SecretStr:
        pwd = v.get_secret_value()
        checks = [
            any(c.isupper() for c in pwd),
            any(c.islower() for c in pwd),
            any(c.isdigit() for c in pwd),
        ]
        if not all(checks):
            raise ValueError(
                "Senha deve conter maiúsculas, minúsculas e números"
            )
        return v


class User(UserBase, TimestampMixin):
    id: UUID = Field(default_factory=uuid4)
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    preferred_genres: list[Genre] = Field(default_factory=list)
    embedding: list[float] | None = None


class UserPublic(UserBase):
    id: UUID
    display_name: str
    preferred_genres: list[Genre]
    created_at: datetime


# ── Interaction ───────────────────────────────────────────────────────────────

INTERACTION_WEIGHTS: dict[InteractionType, float] = {
    InteractionType.PLAY: 1.0,
    InteractionType.SKIP: -0.3,
    InteractionType.LIKE: 3.0,
    InteractionType.DISLIKE: -2.0,
    InteractionType.SAVE: 2.5,
    InteractionType.SHARE: 4.0,
    InteractionType.ADD_TO_PLAYLIST: 3.5,
}


class InteractionCreate(BaseModel):
    user_id: UUID
    track_id: UUID
    interaction_type: InteractionType
    play_duration_ms: int | None = Field(default=None, ge=0)
    context: dict[str, str] = Field(default_factory=dict)


class Interaction(InteractionCreate, TimestampMixin):
    id: UUID = Field(default_factory=uuid4)

    @property
    def weight(self) -> float:
        base = INTERACTION_WEIGHTS[self.interaction_type]
        if (
            self.interaction_type == InteractionType.PLAY
            and self.play_duration_ms is not None
        ):
            return base * min(self.play_duration_ms / 30_000, 1.0)
        return base


# ── Recommendation ────────────────────────────────────────────────────────────

class RecommendationReason(StrEnum):
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    GRAPH_PAGERANK = "graph_pagerank"
    SIMILAR_USERS = "similar_users"
    GENRE_AFFINITY = "genre_affinity"
    HYBRID = "hybrid"


class RecommendedTrack(BaseModel):
    track: Track
    score: float = Field(ge=0.0)
    reason: RecommendationReason
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: UUID
    recommendations: list[RecommendedTrack]
    algorithm_used: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cached: bool = False


class RecommendationRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=50)
    exclude_track_ids: list[UUID] = Field(default_factory=list, max_length=500)
    genres: list[Genre] | None = None
    min_popularity: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: RecommendationReason = RecommendationReason.HYBRID
