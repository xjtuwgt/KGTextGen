import html
import textwrap
from wikigraph.datautils.io_utils import normalize_freebase_string


def truncate(s: str, truncate_limit=100) -> str:
    if len(s) > truncate_limit:
        s = s[:truncate_limit] + '...'
    return s


def format_label(s: str, width: int = 40) -> str:
    """Format a node / edge label."""
    s = normalize_freebase_string(s)
    s = truncate(s)
    lines = s.split('\\n')
    output_lines = []
    for line in lines:
        line = html.escape(line)
        if width > 0:
            output_lines += textwrap.wrap(line, width)
        else:
            output_lines.append(line)
    return '<' + '<br/>'.join(output_lines) + '>'



