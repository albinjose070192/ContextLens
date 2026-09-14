"""
Text cleaning functions for each source type in the corpus.

Source types and their cleaning strategies:
  - doc / readme / changelog : strip markdown + MDX/JSX syntax, keep code content
  - issue                    : concatenate title + body + comments, strip markdown
  - docstring                : prepend node type + name for context, minimal cleaning
  - stackoverflow_qa         : strip HTML, concatenate Q + accepted answer
"""

import re
import html
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Markdown / MDX cleaner
# ---------------------------------------------------------------------------

# Patterns compiled once for performance
_FRONTMATTER     = re.compile(r'^---\s*\n.*?\n---\s*\n', re.DOTALL)
_JS_IMPORT       = re.compile(r'^import\s+.+$', re.MULTILINE)
_JSX_SELFCLOSE   = re.compile(r'<[A-Z][A-Za-z0-9]*[^>]*/>')
_JSX_TAG         = re.compile(r'</?[A-Z][A-Za-z0-9]*[^>]*>')
_HTML_TAG        = re.compile(r'<[^>]+>')
_ADMONITION      = re.compile(r'^:::\s*\w*\s*$', re.MULTILINE)   # VitePress/Docusaurus :::tip etc.
_CODE_FENCE      = re.compile(r'```[\w]*\n?')
_INLINE_CODE     = re.compile(r'`([^`\n]+)`')
_MD_LINK         = re.compile(r'\[([^\]]+)\]\([^\)]+\)')           # [text](url) → text
_MD_IMAGE        = re.compile(r'!\[[^\]]*\]\([^\)]+\)')            # ![alt](url) → ''
_MD_HEADER       = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_BOLD_ITALIC  = re.compile(r'\*{1,3}([^*\n]+)\*{1,3}')
_MD_UNDER_ITALIC = re.compile(r'_{1,3}([^_\n]+)_{1,3}')
_MD_HRULE        = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_MD_BLOCKQUOTE   = re.compile(r'^>\s?', re.MULTILINE)
_MULTI_NEWLINE   = re.compile(r'\n{3,}')
_TRAILING_SPACE  = re.compile(r'[ \t]+$', re.MULTILINE)


def clean_markdown(text: str) -> str:
    """Strip markdown and MDX formatting, keep prose and code content."""
    if not text:
        return ""

    # 1. YAML/TOML frontmatter
    text = _FRONTMATTER.sub('', text)
    # 2. JS/TS import statements (MDX)
    text = _JS_IMPORT.sub('', text)
    # 3. JSX self-closing tags  (<Callout />)
    text = _JSX_SELFCLOSE.sub(' ', text)
    # 4. JSX open/close tags — keep inner text (<Callout>...</Callout> → ...)
    text = _JSX_TAG.sub(' ', text)
    # 5. Remaining HTML tags
    text = _HTML_TAG.sub(' ', text)
    # 6. VitePress/Docusaurus admonition markers (:::tip etc.) — keep body text
    text = _ADMONITION.sub('', text)
    # 7. Code fences — strip fence markers, keep code content
    text = _CODE_FENCE.sub('', text)
    text = text.replace('```', '')
    # 8. Inline code backticks — keep the text
    text = _INLINE_CODE.sub(r'\1', text)
    # 9. Markdown links → link text only
    text = _MD_LINK.sub(r'\1', text)
    # 10. Images → remove entirely
    text = _MD_IMAGE.sub('', text)
    # 11. Headers — strip # prefix, keep heading text
    text = _MD_HEADER.sub('', text)
    # 12. Bold / italic
    text = _MD_BOLD_ITALIC.sub(r'\1', text)
    text = _MD_UNDER_ITALIC.sub(r'\1', text)
    # 13. Horizontal rules
    text = _MD_HRULE.sub('', text)
    # 14. Block quotes — strip > prefix
    text = _MD_BLOCKQUOTE.sub('', text)
    # 15. Trailing whitespace per line
    text = _TRAILING_SPACE.sub('', text)
    # 16. Collapse excess blank lines
    text = _MULTI_NEWLINE.sub('\n\n', text)

    return text.strip()


# ---------------------------------------------------------------------------
# HTML cleaner  (Stack Overflow bodies are rendered HTML)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags while preserving text content and code blocks."""

    _BLOCK_TAGS = {'p', 'div', 'li', 'br', 'tr', 'h1', 'h2', 'h3', 'h4', 'blockquote'}
    _SKIP_TAGS  = {'script', 'style'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip = True
        elif tag in self._BLOCK_TAGS:
            self._buf.append('\n')

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip = False
        elif tag in self._BLOCK_TAGS or tag == 'pre':
            self._buf.append('\n')

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def get_text(self) -> str:
        return ''.join(self._buf)


def clean_html(html_text: str) -> str:
    """Strip HTML tags from Stack Overflow body, keep text and code content."""
    if not html_text:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(html.unescape(html_text))
    text = parser.get_text()
    text = _MULTI_NEWLINE.sub('\n\n', text)
    text = _TRAILING_SPACE.sub('', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Per source-type text builders
# ---------------------------------------------------------------------------

_BOT_PATTERNS = re.compile(
    r'(thanks for (opening|filing|raising)|'
    r'this issue has been automatically|'
    r'please fill (out|in) the (following|template)|'
    r'\*\*describe the bug\*\*|'
    r'\*\*to reproduce\*\*|'
    r'\*\*expected behavior\*\*|'
    r'<!-- .+? -->)',
    re.IGNORECASE,
)


def build_issue_text(artifact: dict) -> str:
    """Concatenate issue title, body, and comments into one text block."""
    parts = [f"Issue: {artifact.get('title', '').strip()}"]

    labels = artifact.get('labels', [])
    if labels:
        parts.append(f"Labels: {', '.join(labels)}")

    body = artifact.get('body', '') or ''
    body = _BOT_PATTERNS.sub('', body).strip()
    if body:
        parts.append(body)

    for comment in artifact.get('comments', []):
        comment = (comment or '').strip()
        comment = _BOT_PATTERNS.sub('', comment).strip()
        if comment:
            parts.append(f"---\nComment:\n{comment}")

    return '\n\n'.join(parts)


def build_docstring_text(artifact: dict) -> str:
    """Prefix docstring with node type and name so chunks are self-contained."""
    node_type = artifact.get('node_type', 'function')
    name      = artifact.get('name', '<module>')
    docstring = (artifact.get('docstring', '') or '').strip()
    return f"[{node_type}] {name}:\n{docstring}"


def build_so_text(artifact: dict) -> str:
    """Concatenate SO question title, question body, and accepted answer."""
    title    = artifact.get('title', '').strip()
    q_body   = clean_html(artifact.get('question_body', '') or '')
    a_body   = clean_html(artifact.get('answer_body', '') or '')

    parts = []
    if title:
        parts.append(f"Question: {title}")
    if q_body:
        parts.append(q_body)
    if a_body:
        parts.append(f"Accepted Answer:\n{a_body}")

    return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_text(artifact: dict) -> str:
    """Return cleaned plain text for any artifact, dispatching on source_type."""
    st = artifact.get('source_type', '')

    if st in ('doc', 'readme', 'changelog'):
        return clean_markdown(artifact.get('content', '') or '')

    if st == 'issue':
        raw = build_issue_text(artifact)
        return clean_markdown(raw)

    if st == 'docstring':
        return build_docstring_text(artifact)  # already clean

    if st == 'stackoverflow_qa':
        return build_so_text(artifact)  # HTML cleaned inside builder

    return ''
