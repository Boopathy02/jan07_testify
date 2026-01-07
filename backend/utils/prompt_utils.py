# utils/prompt_utils.py
import os
from pathlib import Path
from typing import List, Optional

def get_prompt(prompt_name: str, project_src_dir: Optional[Path]) -> str:
    """
    Gets a prompt, prioritizing the project-specific one if it exists,
    otherwise falling back to the global prompt.
    """
    # 1. Try to find the prompt in the active project's directory
    if project_src_dir:
        project_prompt_path = project_src_dir / "prompts" / prompt_name
        if project_prompt_path.exists():
            return project_prompt_path.read_text(encoding="utf-8")

    # 2. Fallback to the global prompts directory
    global_prompt_path = Path(__file__).resolve().parent.parent / "prompts" / prompt_name
    if not global_prompt_path.exists():
        raise FileNotFoundError(f"Prompt '{prompt_name}' not found in project or global directories.")
        
    return global_prompt_path.read_text(encoding="utf-8")

def build_prompt(
    story_block: str,
    method_map: dict,
    page_names: list[str],
    site_url: str,
    dynamic_steps: list[str],
    project_src_dir: Path,
) -> str:
    def _normalize_method(m: str) -> str:
        m_norm = m.strip()
        if m_norm.startswith("def "):
            m_norm = m_norm[4:].strip()
        return m_norm

    page_method_section = "\n".join(
        f"# {p}:\n" + "\n".join(f"- def {_normalize_method(m)}" for m in method_map.get(p, []))
        for p in page_names
    )
    dynamic_steps_joined = "\n".join(dynamic_steps)
    site_url_escaped = site_url.replace('"', '"')

    template = get_prompt("ui_test_generation.txt", project_src_dir)
    return template.format(
        story_block=story_block,
        site_url=site_url_escaped,
        page_method_section=page_method_section,
        dynamic_steps=dynamic_steps_joined,
    )

def build_security_prompt(
    story_block: str,
    method_map: dict,
    page_names: list[str],
    site_url: str,
    project_src_dir: Path,
    security_matrix: str = "",
) -> str:
    def _normalize_method(m: str) -> str:
        m_norm = m.strip()
        if m_norm.startswith("def "):
            m_norm = m_norm[4:].strip()
        return m_norm

    page_method_section = "\n".join(
        f"# {p}:\n" + "\n".join(f"- def {_normalize_method(m)}" for m in method_map.get(p, []))
        for p in page_names
    )
    input_methods = []
    prefixes = ("enter_", "fill_", "type_", "set_", "input_", "select_", "choose_", "add_", "pick_", "search_", "upload_", "set_value_", "type_in_")
    keywords = ("input", "field", "email", "password", "username", "name", "phone", "address", "zip", "code", "value", "text")
    for _page, methods in method_map.items():
        for method in methods:
            method_name = method.split("(", 1)[0].replace("def ", "").strip()
            if method_name.startswith(prefixes) or any(k in method_name for k in keywords):
                input_methods.append(method_name)
    payload_list = "\n".join(
        f"- {m}(page, <payload>)" for m in sorted(set(input_methods))
    )
    site_url_escaped = site_url.replace('"', '"')

    if not (security_matrix or "").strip():
        template = get_prompt("security_test_generation.txt", project_src_dir)
        return template.format(
            story_block=story_block,
            page_method_section=page_method_section,
            payload_list=payload_list,
            site_url=site_url_escaped,
        )
    else:
        template = get_prompt("security_test_generation_matrix.txt", project_src_dir)
        return template.format(
            story_block=story_block,
            security_matrix=security_matrix,
            site_url=site_url_escaped,
            page_method_section=page_method_section,
            payload_list=payload_list,
        )

def build_accessibility_prompt(
    story_block: str,
    method_map: dict,
    page_names: list[str],
    site_url: str,
    project_src_dir: Path,
) -> str:
    def _normalize_method(m: str) -> str:
        m_norm = m.strip()
        if m_norm.startswith("def "):
            m_norm = m_norm[4:].strip()
        return m_norm

    page_method_section = "\n".join(
        f"# {p}:\n" + "\n".join(f"- def {_normalize_method(m)}" for m in method_map.get(p, []))
        for p in page_names
    )

    template = get_prompt("accessibility_test_generation.txt", project_src_dir)
    return template.format(
        story_block=story_block,
        page_method_section=page_method_section,
    )
