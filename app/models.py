from app.db import Base
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey,
    Text, Float, Boolean, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)

    role = Column(String(20), nullable=False, default="HR")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("role IN ('HR', 'ADMIN')", name="check_user_role"),
    )

    uploads = relationship("Upload", back_populates="user", cascade="all, delete-orphan")


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="uploads")
    analyses = relationship("Analysis", back_populates="upload", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    upload_id = Column(
        Integer,
        ForeignKey("uploads.id", ondelete="CASCADE"),
        nullable=False
    )

    profile = Column(String(100), nullable=False, default="IT")
    weights_json = Column(Text, nullable=False)

    alpha = Column(Float, nullable=False, default=0.7)
    created_at = Column(DateTime, default=datetime.utcnow)
    summary_json = Column(Text, nullable=True)

    upload = relationship("Upload", back_populates="analyses")

    __table_args__ = (
        CheckConstraint("alpha >= 0 AND alpha <= 1", name="check_alpha_range"),
    )


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    criteria = relationship(
        "ProfileCriterion",
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="ProfileCriterion.sort_order"
    )


class ProfileCriterion(Base):
    __tablename__ = "profile_criteria"

    id = Column(Integer, primary_key=True, index=True)

    profile_id = Column(
        Integer,
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    code = Column(String(10), nullable=False)     # C1..C6
    label = Column(String(150), nullable=False)

    weight = Column(Float, nullable=False, default=0.0)
    sort_order = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)

    profile = relationship("Profile", back_populates="criteria")

    __table_args__ = (
        UniqueConstraint("profile_id", "code", name="uq_profile_code"),
        CheckConstraint("weight >= 0", name="check_weight_positive"),
    )