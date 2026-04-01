"""Skill configuration endpoints for user-level skill preferences."""

import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import zipfile

import structlog
import yaml
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..core.auth_deps import get_current_user
from ..models import User

logger = structlog.get_logger(__name__)

router = APIRouter()

# ---- constants ----

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_SKILL_NAME_LEN = 64
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# ---- helpers ----

def _skill_config_path(user_id: str) -> Path:
    base = os.environ.get("BASE_DATA_DIR", "/data")
    return Path(base) / "memories" / user_id / "skill_config.json"


def _user_skills_dir(user_id: str) -> Path:
    base = os.environ.get("BASE_DATA_DIR", "/data")
    return Path(base) / "skills" / user_id


def _system_skills_dir() -> Path:
    qsa_root = os.environ.get("QSA_ROOT", "/app/quantitative_strategy_agent/qsa")
    return Path(qsa_root) / "skills" / "system"


def _parse_skill_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from SKILL.md content."""
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


def _scan_skills_dir(skills_dir: Path, source: str) -> list[dict]:
    """Scan a skills directory and return metadata for each skill."""
    result = []
    if not skills_dir.is_dir():
        return result
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        md_file = skill_dir / "SKILL.md"
        if not md_file.is_file():
            continue
        meta = {}
        try:
            content = md_file.read_text(encoding="utf-8")
            meta = _parse_skill_frontmatter(content)
        except Exception:
            pass
        result.append({
            "slug": skill_dir.name,                       # 改 "id" → "slug"
            "name": meta.get("name", skill_dir.name),
            "description": meta.get("description", ""),
            "source": source,                             # 改 "type" → "source"
        })
    return result


def _list_all_skills(user_id: str) -> list[dict]:
    """List system + user skills."""
    system = _scan_skills_dir(_system_skills_dir(), "system")
    user = _scan_skills_dir(_user_skills_dir(user_id), "user")
    return system + user


def _read_enabled_skills(user_id: str) -> list[str] | None:
    cfg_path = _skill_config_path(user_id)
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f).get("enabled_skills")
    except Exception:
        return None


def _validate_skill_name(name: str) -> str | None:
    """Return error message if name is invalid, else None."""
    if not name:
        return "name is required in YAML frontmatter"
    if len(name) > MAX_SKILL_NAME_LEN:
        return f"name must be <= {MAX_SKILL_NAME_LEN} characters"
    if not SKILL_NAME_RE.match(name):
        return "name must match ^[a-z0-9]+(-[a-z0-9]+)*$"
    return None


# ---- models ----

class SkillConfigUpdate(BaseModel):
    enabled_skills: list[str]


class SkillInfo(BaseModel):
    slug: str
    name: str
    description: str
    source: str
    enabled: bool

class SkillConfigResponse(BaseModel):
    enabled_skills: list[str] | None = None
    available_skills: list[SkillInfo] = []

class SkillUploadResponse(BaseModel):
    slug: str
    name: str
    description: str
    source: str
    created_at: str


# ---- endpoints ----

@router.get("/api/user/skill-config", response_model=SkillConfigResponse)
async def get_skill_config(user: User = Depends(get_current_user)):
    """Get current user's skill configuration with full skill list."""
    cfg_path = _skill_config_path(user.identity)
    enabled_skills = None
    updated_at = None

    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                data = json.load(f)
            enabled_skills = data.get("enabled_skills")
            updated_at = data.get("updated_at")
        except Exception:
            logger.warning(f"Failed to read skill config: {cfg_path}", exc_info=True)

    all_skills = _list_all_skills(user.identity)
    skill_infos = []
    for s in all_skills:
        is_enabled = enabled_skills is None or s["slug"] in enabled_skills   # 改 s["id"] → s["slug"]
        skill_infos.append(SkillInfo(
            slug=s["slug"],              # 改 id=s["id"]
            name=s["name"],
            description=s["description"],
            source=s["source"],          # 改 type=s["type"]  (但实际 _scan 里用的是 "source" 了)
            enabled=is_enabled,
        ))

    return SkillConfigResponse(
        enabled_skills=enabled_skills,
        updated_at=updated_at,
        available_skills=skill_infos,
    )


@router.put("/api/user/skill-config", response_model=SkillConfigResponse)
async def update_skill_config(
    body: SkillConfigUpdate,
    user: User = Depends(get_current_user),
):
    """Update current user's enabled skills list."""
    cfg_path = _skill_config_path(user.identity)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "enabled_skills": body.enabled_skills,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    with open(cfg_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"[update_skill_config] user={user.identity} enabled_skills={body.enabled_skills}"
    )

    all_skills = _list_all_skills(user.identity)
    skill_infos = []
    for s in all_skills:
        skill_infos.append(SkillInfo(
            slug=s["slug"],              # 改 id=s["id"]
            name=s["name"],
            description=s["description"],
            source=s["source"],          # 改
            enabled=s["slug"] in data["enabled_skills"],   # 改 s["id"]
        ))

    return SkillConfigResponse(
        enabled_skills=data["enabled_skills"],
        updated_at=data["updated_at"],
        available_skills=skill_infos,
    )

def _find_skill_md(extract_root: Path) -> Path | None:
    """Find SKILL.md in extracted zip — at root or in a single subdirectory."""
    # 1. 直接在根目录
    candidate = extract_root / "SKILL.md"
    if candidate.is_file():
        return candidate
    # 2. 唯一子目录中
    subdirs = [d for d in extract_root.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        candidate = subdirs[0] / "SKILL.md"
        if candidate.is_file():
            return candidate
    # 3. 递归搜索（仅一层子目录）
    for subdir in subdirs:
        candidate = subdir / "SKILL.md"
        if candidate.is_file():
            return candidate
    return None

@router.post("/api/skills/upload", response_model=SkillUploadResponse)
async def upload_skill(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Upload a custom SKILL.md or .zip containing SKILL.md."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    filename_lower = file.filename.lower()
    is_zip = filename_lower.endswith(".zip")
    is_md = filename_lower.endswith(".md")

    if not is_zip and not is_md:
        raise HTTPException(400, "File must be a .md or .zip file")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_SIZE // 1024 // 1024} MB)")

    if is_md:
        # ---- 单文件 .md 上传 (现有逻辑) ----
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be valid UTF-8 text")

        meta = _parse_skill_frontmatter(text)
        name = meta.get("name", "")
        err = _validate_skill_name(name)
        if err:
            raise HTTPException(422, f"Invalid skill: {err}")

        slug = name
        system_dir = _system_skills_dir() / slug
        if system_dir.is_dir():
            raise HTTPException(409, f"Cannot override system skill '{slug}'.")

        user_dir = _user_skills_dir(user.identity)
        skill_dir = user_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")

    else:
        # ---- .zip 上传 ----
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "upload.zip"
            zip_path.write_bytes(content)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    # 安全检查：禁止路径穿越
                    for info in zf.infolist():
                        if info.filename.startswith("/") or ".." in info.filename:
                            raise HTTPException(400, f"Unsafe path in zip: {info.filename}")
                    zf.extractall(tmp_path / "extracted")
            except zipfile.BadZipFile:
                raise HTTPException(400, "Invalid zip file")

            extract_root = tmp_path / "extracted"

            # 找 SKILL.md — 直接在根目录或唯一子目录中
            skill_md_path = _find_skill_md(extract_root)
            if skill_md_path is None:
                raise HTTPException(422, "Zip must contain a SKILL.md file (at root or in a single subdirectory)")

            text = skill_md_path.read_text(encoding="utf-8")
            meta = _parse_skill_frontmatter(text)
            name = meta.get("name", "")
            err = _validate_skill_name(name)
            if err:
                raise HTTPException(422, f"Invalid skill: {err}")

            slug = name
            system_dir = _system_skills_dir() / slug
            if system_dir.is_dir():
                raise HTTPException(409, f"Cannot override system skill '{slug}'.")

            # 将 SKILL.md 所在目录的所有内容拷贝到目标
            source_dir = skill_md_path.parent
            user_dir = _user_skills_dir(user.identity)
            skill_dir = user_dir / slug

            # 如果已存在先清理
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
            shutil.copytree(source_dir, skill_dir)

    logger.info(f"[upload_skill] user={user.identity} skill={slug} path={skill_dir}")

    return SkillUploadResponse(
        slug=slug,
        name=meta.get("name", slug),
        description=meta.get("description", ""),
        source="user",
        created_at=datetime.now(UTC).isoformat(),
    )


@router.delete("/api/skills/{slug}")
async def delete_skill(
    slug: str,
    user: User = Depends(get_current_user),
):
    """Delete a user-uploaded custom skill."""
    # Validate slug to prevent path traversal
    if not SKILL_NAME_RE.match(slug) or ".." in slug or "/" in slug:
        raise HTTPException(400, "Invalid skill slug")

    # Cannot delete system skills
    system_dir = _system_skills_dir() / slug
    if system_dir.is_dir():
        raise HTTPException(403, "Cannot delete system skills")

    user_dir = _user_skills_dir(user.identity)
    skill_dir = user_dir / slug

    if not skill_dir.is_dir():
        raise HTTPException(404, f"Skill '{slug}' not found")

    shutil.rmtree(skill_dir)

    # Also remove from enabled_skills if present
    cfg_path = _skill_config_path(user.identity)
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            enabled = cfg.get("enabled_skills")
            if enabled and slug in enabled:
                enabled.remove(slug)
                cfg["updated_at"] = datetime.now(UTC).isoformat()
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        except Exception:
            pass

    logger.info(f"[delete_skill] user={user.identity} skill={slug}")
    return {"deleted": slug}

@router.get("/api/skills/{slug}/content")
async def get_skill_content(
    slug: str,
    user: User = Depends(get_current_user),
):
    """Get the full SKILL.md content for a skill (system or user)."""
    if not SKILL_NAME_RE.match(slug) or ".." in slug or "/" in slug:
        raise HTTPException(400, "Invalid skill slug")

    # 先查用户 skill，再查系统 skill
    user_md = _user_skills_dir(user.identity) / slug / "SKILL.md"
    system_md = _system_skills_dir() / slug / "SKILL.md"

    md_path = user_md if user_md.is_file() else system_md if system_md.is_file() else None
    if md_path is None:
        raise HTTPException(404, f"Skill '{slug}' not found")

    return {"slug": slug, "content": md_path.read_text(encoding="utf-8")}