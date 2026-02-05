"""
Post-processors that convert raw OCR output (XML, Markdown, plain text)
into the JSON structures expected by benchmark scoring functions.

medieval_manuscripts expects:
    {"folios": [{"folio": str, "text": str, "addition1": str, ...}]}

fraktur_adverts expects:
    {"advertisements": [{"date": str|None, "tags_section": str|None, "text": str}]}
"""

import logging
import re
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Churro XML → Benchmark JSON
# ---------------------------------------------------------------------------

def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix: '{http://...}Heading' → 'heading'."""
    if tag and "}" in tag:
        return tag.split("}", 1)[1].lower()
    return (tag or "").lower()


def _strip_xml_tags(xml_text: str) -> str:
    """Remove all XML tags and return plain text."""
    return re.sub(r"<[^>]+>", "", xml_text).strip()


def _try_parse_churro_xml(raw: str) -> ET.Element | None:
    """Try to parse Churro XML output, handling common issues."""
    raw = raw.strip()
    # Churro sometimes outputs text before the XML root
    xml_start = raw.find("<")
    if xml_start > 0:
        raw = raw[xml_start:]

    # Strip XML namespaces to avoid {uri}Tag prefixes on every element
    raw = re.sub(r'\s+xmlns\s*=\s*"[^"]*"', "", raw)
    raw = re.sub(r'\s+xmlns:[a-zA-Z]+\s*=\s*"[^"]*"', "", raw)

    # Wrap in a root if there are multiple top-level elements
    if not raw.startswith("<?xml"):
        # Check if already has single root
        try:
            return ET.fromstring(raw)
        except ET.ParseError:
            pass
        # Try wrapping
        try:
            return ET.fromstring(f"<root>{raw}</root>")
        except ET.ParseError as e:
            log.warning(f"XML parse failed (wrapped): {e} — falling back to text extraction")
            return None
    else:
        try:
            return ET.fromstring(raw)
        except ET.ParseError:
            # Strip XML declaration and try again
            raw = re.sub(r"<\?xml[^>]*\?>", "", raw).strip()
            try:
                return ET.fromstring(f"<root>{raw}</root>")
            except ET.ParseError as e:
                log.warning(f"XML parse failed (after decl strip): {e} — falling back to text extraction")
                return None


def _collect_text(element: ET.Element) -> str:
    """Recursively collect all text content from an element."""
    parts = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        parts.append(_collect_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return "\n".join(p for p in parts if p)


def _find_elements_recursive(root: ET.Element, tag: str) -> list[ET.Element]:
    """Find all elements with given tag name (case-insensitive, namespace-aware) recursively."""
    tag_lower = tag.lower()
    results = []
    if _local_tag(root.tag) == tag_lower:
        results.append(root)
    for child in root:
        results.extend(_find_elements_recursive(child, tag))
    return results


def churro_xml_to_medieval(raw: str) -> dict:
    """
    Convert Churro XML output to medieval_manuscripts JSON format.

    Mapping:
        <Header><PageNumber> or <Footer><FolioNumber> → folio
        <Body><Paragraph><Line> elements                → text
        <MarginalNote> elements (1st, 2nd, ...)          → addition1, addition2, ...
    """
    root = _try_parse_churro_xml(raw)
    if root is None:
        # Fallback: strip XML tags, return as single folio
        plain = _strip_xml_tags(raw) if "<" in raw else raw
        return {"folios": [{"folio": "", "text": plain, "addition1": ""}]}

    folios = []

    # Look for Page elements (Churro wraps each page in <Page>)
    pages = _find_elements_recursive(root, "Page")
    if not pages:
        # No <Page> wrappers — treat entire output as one folio
        pages = [root]

    for page in pages:
        folio_num = ""
        body_lines = []
        marginal_notes = []

        # Extract folio number from Header/PageNumber or Footer/FolioNumber
        for tag in ["PageNumber", "FolioNumber", "Pagenumber", "Folionumber"]:
            elems = _find_elements_recursive(page, tag)
            if elems:
                text = _collect_text(elems[0]).strip()
                # Strip brackets: "[3r]" → "3r", "[70v]" → "70v"
                text = text.strip("[]() ")
                # Also strip trailing letters like 'r' and 'v' for folio number
                # The ground truth uses just the number: "3" not "3r"
                num_match = re.match(r"(\d+)", text)
                if num_match:
                    folio_num = num_match.group(1)
                else:
                    folio_num = text
                break

        # Extract body text from Body/Paragraph/Line elements
        body_elems = _find_elements_recursive(page, "Body")
        if body_elems:
            for body in body_elems:
                paragraphs = _find_elements_recursive(body, "Paragraph")
                if paragraphs:
                    for para in paragraphs:
                        lines = _find_elements_recursive(para, "Line")
                        if lines:
                            for line in lines:
                                text = _collect_text(line).strip()
                                if text:
                                    body_lines.append(text)
                        else:
                            text = _collect_text(para).strip()
                            if text:
                                body_lines.append(text)
                else:
                    text = _collect_text(body).strip()
                    if text:
                        body_lines.append(text)
        else:
            # No <Body> found — collect all non-marginal, non-header text
            for child in page:
                lt = _local_tag(child.tag)
                if lt not in ("marginalnote", "header", "footer",
                              "pagenumber", "folionumber", "metadata"):
                    text = _collect_text(child).strip()
                    if text:
                        body_lines.append(text)

        # Extract marginal notes
        notes = _find_elements_recursive(page, "MarginalNote")
        for note in notes:
            text = _collect_text(note).strip()
            marginal_notes.append(text)

        # Build folio entry
        folio_entry = {
            "folio": folio_num,
            "text": "\n".join(body_lines),
        }
        for i, note_text in enumerate(marginal_notes, 1):
            folio_entry[f"addition{i}"] = note_text

        # Ensure at least addition1 exists
        if "addition1" not in folio_entry:
            folio_entry["addition1"] = ""

        folios.append(folio_entry)

    if not folios:
        plain = _strip_xml_tags(raw) if "<" in raw else raw
        return {"folios": [{"folio": "", "text": plain, "addition1": ""}]}

    return {"folios": folios}


def _normalize_section_name(text: str) -> str:
    """Normalize a section heading for better matching against ground truth.

    Churro faithfully transcribes archaic characters like long-s (ſ) that
    differ from the modern spellings used in ground truth section names.
    Since section names are only used as grouping keys (not scored for text
    quality), normalizing them improves matching without inflating scores.
    """
    # Long-s → modern s (ſ is just an archaic glyph for s)
    text = text.replace("ſ", "s")
    # Strip trailing punctuation that varies between OCR and ground truth
    text = text.rstrip(":;,. ")
    return text


def churro_xml_to_fraktur(raw: str) -> dict:
    """
    Convert Churro XML output to fraktur_adverts JSON format.

    Strategy:
        1. Walk through <Body> elements sequentially
        2. When encountering a <Heading>, set as current tags_section
        3. Concatenate <Line> text, then split on numbered prefixes (1., 2., ...)
           to segment individual ads
    """
    root = _try_parse_churro_xml(raw)
    if root is None:
        plain = _strip_xml_tags(raw) if "<" in raw else raw
        return _text_to_fraktur(plain)

    # Collect all text in document order, preserving headings
    sections = []
    current_section = ""
    current_lines = []

    def _walk(elem):
        nonlocal current_section, current_lines

        lt = _local_tag(elem.tag)

        if lt in ("heading", "title", "sectionheading"):
            # Flush current lines
            if current_lines:
                sections.append((current_section, "\n".join(current_lines)))
                current_lines = []
            current_section = _normalize_section_name(_collect_text(elem).strip())
            return

        if lt in ("line", "item"):
            text = _collect_text(elem).strip()
            if text:
                current_lines.append(text)
            return

        # Skip metadata blocks entirely
        if lt in ("metadata",):
            return

        # Recurse into children
        if elem.text and elem.text.strip() and lt not in (
            "page", "body", "paragraph", "list", "root", "document",
            "historicaldocument", "header", "footer", "figure",
            "blockquotation",
        ):
            current_lines.append(elem.text.strip())

        for child in elem:
            _walk(child)

    _walk(root)
    if current_lines:
        sections.append((current_section, "\n".join(current_lines)))

    # Now split each section's text into individual ads by numbered prefixes
    advertisements = []
    for section_name, section_text in sections:
        ads = _split_numbered_ads(section_text)
        for ad_text in ads:
            advertisements.append({
                "date": None,
                "tags_section": section_name if section_name else None,
                "text": ad_text.strip(),
            })

    if not advertisements:
        plain = _strip_xml_tags(raw) if "<" in raw else raw
        return _text_to_fraktur(plain)

    return {"advertisements": advertisements}


# ---------------------------------------------------------------------------
# Plain text / Markdown → Benchmark JSON
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting."""
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove images ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove block quotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text.strip()


def _split_numbered_ads(text: str) -> list[str]:
    """
    Split text into individual advertisements by numbered prefixes.

    Matches patterns like "1. ", "2. ", "16. " at line beginnings or after
    paragraph breaks. Returns list of ad texts, each starting with its number.
    """
    # Pattern: number followed by period and space at start of line (MULTILINE)
    parts = re.split(r"(?=^\s*\d+\.\s)", text.strip(), flags=re.MULTILINE)
    ads = []
    for part in parts:
        part = part.strip()
        if part:
            ads.append(part)
    return ads


def text_to_medieval(raw: str) -> dict:
    """Convert plain text / Markdown OCR output to medieval_manuscripts format."""
    text = _strip_markdown(raw) if raw.startswith("#") or "**" in raw else raw
    text = text.strip()
    return {
        "folios": [{
            "folio": "",
            "text": text,
            "addition1": "",
        }]
    }


def _text_to_fraktur(text: str) -> dict:
    """Convert plain text to fraktur_adverts format by splitting on numbered ads."""
    text = text.strip()
    if not text:
        return {"advertisements": []}

    ads = _split_numbered_ads(text)
    advertisements = []
    for ad_text in ads:
        advertisements.append({
            "date": None,
            "tags_section": None,
            "text": ad_text.strip(),
        })

    if not advertisements:
        # No numbered pattern found — return entire text as one ad
        advertisements.append({
            "date": None,
            "tags_section": None,
            "text": text,
        })

    return {"advertisements": advertisements}


def text_to_fraktur(raw: str) -> dict:
    """Convert plain text / Markdown OCR output to fraktur_adverts format."""
    text = _strip_markdown(raw) if raw.startswith("#") or "**" in raw else raw
    return _text_to_fraktur(text)


# ---------------------------------------------------------------------------
# Public API: route model output to the correct converter
# ---------------------------------------------------------------------------

# Converter dispatch table: (output_format, benchmark_name) → function
CONVERTERS = {
    ("xml", "medieval_manuscripts"): churro_xml_to_medieval,
    ("xml", "fraktur_adverts"): churro_xml_to_fraktur,
    ("markdown", "medieval_manuscripts"): text_to_medieval,
    ("markdown", "fraktur_adverts"): text_to_fraktur,
    ("text", "medieval_manuscripts"): text_to_medieval,
    ("text", "fraktur_adverts"): text_to_fraktur,
}


def convert(output_format: str, benchmark_name: str, raw_text: str) -> dict:
    """
    Convert raw model output to the benchmark's expected JSON structure.

    Args:
        output_format: One of "xml", "markdown", "text"
        benchmark_name: e.g. "medieval_manuscripts", "fraktur_adverts"
        raw_text: The raw OCR output string

    Returns:
        Parsed dict matching the benchmark's expected format
    """
    key = (output_format, benchmark_name)
    converter = CONVERTERS.get(key)
    if converter is None:
        raise ValueError(
            f"No converter for format={output_format!r}, benchmark={benchmark_name!r}. "
            f"Available: {list(CONVERTERS.keys())}"
        )
    return converter(raw_text)
