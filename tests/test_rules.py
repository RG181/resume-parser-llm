from datetime import date

import pytest

from resume_parser.cleaning import clean_text
from resume_parser.dates import RANGE_RE, month_range, parse_date, total_months
from resume_parser.privacy import redact_contact
from resume_parser.rule_extractors import extract_email, extract_name, extract_phone
from resume_parser.segmentation import heading_of, segment
from resume_parser.taxonomy import SkillTaxonomy
from resume_parser.config import load_config, resolve_path
from tests.helpers import SAMPLES

TODAY = date(2026, 10, 3)


@pytest.fixture(scope="module")
def tax():
    cfg = load_config()
    return SkillTaxonomy.from_file(resolve_path(cfg, cfg["pipeline"]["skills_file"]))


# ---------------------------------------------------------------- contact
def test_email():
    assert extract_email("Reach me: Riya.S@Example.com today") == "riya.s@example.com"
    assert extract_email("no email here") is None


@pytest.mark.parametrize("text,expected", [
    ("Call +91-98765 43210 now", "+91-98765 43210"),
    ("Mobile: 09123456780", "09123456780"),
    ("(415) 555-2671", "(415) 555-2671"),
])
def test_phone(text, expected):
    assert extract_phone(text) == expected


def test_year_ranges_are_not_phones():
    assert extract_phone("2019 - 2021 2022 - 2023") is None
    assert extract_phone("CGPA 8.7 batch 2022-2026") is None


def test_name():
    assert extract_name(["RIYA SHARMA", "Jaipur | a@b.com"]) == "Riya Sharma"
    assert extract_name(["Resume", "Arjun Mehta"]) == "Arjun Mehta"
    assert extract_name(["EDUCATION", "x@y.com"]) is None


# ---------------------------------------------------------------- dates
@pytest.mark.parametrize("text", ["Jun 2024 - Aug 2024", "2021–2023", "Jan '22 - Present", "03/2022 to 05/2023",
                                  "September 2020 - March 2021"])
def test_range_detected(text):
    assert RANGE_RE.search(text)


def test_marketing_is_not_a_month():
    m = RANGE_RE.search("marketing 2020 - 2021 team")
    assert m.group(1) == "2020"  # range starts at the year; "mar" inside "marketing" is not a month
    assert not RANGE_RE.search("marketing 2020 team")


def test_month_arithmetic():
    assert month_range("Jun 2024", "Aug 2024", TODAY) == (2024 * 12 + 5, 2024 * 12 + 7)
    assert parse_date("Present", True, TODAY) == 2026 * 12 + 9
    assert parse_date("2024-06") == 2024 * 12 + 5
    assert parse_date("Jan '22") == 2022 * 12
    assert parse_date("2026", True, TODAY) == 2026 * 12 + 9  # future year-end capped at today
    assert parse_date("garbage") is None


def test_overlapping_roles_not_double_counted():
    a = month_range("Jan 2024", "Jun 2024")
    b = month_range("Apr 2024", "Aug 2024")  # overlaps a
    c = month_range("Jan 2025", "Mar 2025")
    assert total_months([a, b, c]) == 8 + 3
    assert month_range("Aug 2024", "Jan 2024") is None  # end before start is rejected


# ---------------------------------------------------------------- cleaning / segmentation
def test_clean_text():
    out = clean_text("• Built  a tool\r\nPage 2 of 3\n\n\n\nco-\noperate\u200b")
    assert out.startswith("- Built a tool")
    assert "Page 2" not in out and "cooperate" in out and "\n\n\n" not in out


@pytest.mark.parametrize("line,section", [("EDUCATION", "education"), ("Work Experience:", "experience"),
                                          ("Technical Proficiency", "skills"), ("Educaton", "education"),
                                          ("Academic Projects", "projects")])
def test_headings(line, section):
    assert heading_of(line) == section


@pytest.mark.parametrize("line", ["Languages: Python, SQL, Java, C++", "- Python", "I enjoy education systems a lot"])
def test_not_headings(line):
    assert heading_of(line) is None


def test_segment_sample():
    sections = segment(clean_text((SAMPLES / "resume_01.txt").read_text(encoding="utf-8")))
    assert {"header", "education", "experience", "skills", "projects"} <= set(sections)
    assert segment("just one blob of text without headings") == {"full": "just one blob of text without headings"}


# ---------------------------------------------------------------- skills
def test_skill_boundaries(tax):
    found = tax.find("Java, JavaScript, C++, C#, Node.js and R. We go to market. DistilBERT")
    assert {"Java", "JavaScript", "C++", "C#", "Node.js", "R"} <= set(found)
    assert "C" not in found and "BERT" not in found and "Go" not in found


def test_skill_aliases(tax):
    assert tax.normalize("sklearn") == "scikit-learn"
    assert tax.normalize("JS") == "JavaScript"
    assert tax.normalize("SomethingNew") == "SomethingNew"
    assert set(tax.find("used sklearn and k8s")) == {"scikit-learn", "Kubernetes"}
    assert tax.category("Docker") == "tool"


# ---------------------------------------------------------------- privacy
def test_redaction():
    out = redact_contact("a.b@c.com | +91-98765 43210 | linkedin.com/in/x | 2020 - 2024")
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]" in out and "2020 - 2024" in out


def test_js_alias_regression(tax):
    """Found by error analysis: 'JS' was never matched, while the 'js' in 'Node.js' was a false positive."""
    assert "JavaScript" in tax.find("Skills: JS, React")
    assert "JavaScript" not in tax.find("Backend in Node.js and Express")
    assert tax.normalize("js") == "JavaScript"


# ---- regressions found on the public ATS dataset / real PDFs
def test_flattened_text_gets_line_breaks_back():
    flat = ("Jane Roe Data Analyst  Pune, India  " + "filler words here " * 30 + "  WORK EXPERIENCE  Analyst  Acme  "
            "Jan 2020 to Mar 2021  EDUCATION  B.Tech  Some University  2019  SKILLS  Python  SQL  Excel  Tableau")
    sections = segment(clean_text(flat))
    assert {"experience", "education", "skills"} <= set(sections)


def test_flat_text_threshold_leaves_normal_text_alone():
    normal = "Line one\nLine two  has a double space\nLine three"
    assert clean_text(normal).count("\n") == 2


def test_glued_heading_and_lone_bullets():
    text = "Michael Smith\nWORK EXPERIENCESoftware Engineer\nMicrosoft\nJan 2020 to Mar 2021\nEDUCATION\nBSc\n"
    s = segment(clean_text(text))
    assert s["experience"].startswith("Software Engineer") and "education" in s
    assert "SKILLSET" not in segment(clean_text("A B\nSKILLSET\nPython")).get("skills", "")  # not split mid-word
    assert clean_text("•\nBuilt a thing\n•\nShipped it") == "- Built a thing\n- Shipped it"


# ---- regression tests for the layouts found on a real-world LaTeX resume (institute above degree, wrapped bullets, ...)
def test_institute_above_degree_and_school_headings():
    from resume_parser.rule_extractors import extract_education
    text = ("Manipal University Jaipur | Jaipur, Rajasthan\nBachelor of Technology in Data Science | Aug 2023 – May 2027\n"
            "- CGPA: 9.41 / 10.0 | 7th Semester\n"
            "Amicus International School | Bharuch, Gujarat\nClass XII – CBSE | 85.0% | 2022 – 2023\n"
            "Queen of Angels Convent Higher Secondary School | Bharuch, Gujarat\nClass X – ICSE | 94.0% | 2020 – 2021")
    edu = extract_education(text)
    assert [e.institute for e in edu] == ["Manipal University Jaipur", "Amicus International School",
                                         "Queen of Angels Convent Higher Secondary School"]
    assert (edu[0].degree, edu[0].field_of_study, edu[0].year) == ("Bachelor of Technology", "Data Science", "2027")
    assert edu[1].score == "85.0%"


def test_experience_company_on_next_line_and_wrapped_bullets():
    from resume_parser.rule_extractors import extract_experience
    text = ("Software Engineering Program Intern | May 2026 – Jul 2026\nJ.P. Morgan Chase & Co. | Remote\n"
            "- Selected through a national program for the Software Engineering\nProgram Internship (18 May – 10 July 2026); completed it.\n"
            "HR Lead, Autonomous Initiative Club | Aug 2024 – May 2025\nManipal University Jaipur | Jaipur\n- Managed club operations.")
    exp = extract_experience(text)
    assert len(exp) == 2 and exp[0].company == "J.P. Morgan Chase & Co." and exp[0].role == "Software Engineering Program Intern"
    assert len(exp[0].highlights) == 1  # the wrapped line was glued to its bullet, not treated as a new role
    assert exp[1].role == "HR Lead" and "Autonomous Initiative Club" in exp[1].company


def test_bullet_sentence_is_not_mistaken_for_company():
    from resume_parser.rule_extractors import extract_experience
    text = "Data Analyst Intern | Jun 2023 – Aug 2023\nCleaned and analysed datasets with Django.\n"
    assert extract_experience(text)[0].company is None


def test_projects_wrapped_bullets_tech_stack_and_context():
    from resume_parser.rule_extractors import extract_projects
    text = ("ExpressScan | React Native, Firebase | Walmart Sparkathon\n- Built a smart retail app enabling customers to scan\n"
            "barcodes and pay via UPI.\n- Engineered a real-time 3D store model.\n"
            "AI Search | 3rd Place, InnovateX 2.0\n- Tech Stack: Flask, HuggingFace Transformers (DistilBART, BART MNLI), Web Speech API")
    p = extract_projects(text)
    assert [x.name for x in p] == ["ExpressScan", "AI Search"]
    assert p[0].tech == ["React Native", "Firebase"] and p[0].context == "Walmart Sparkathon"
    assert p[1].context == "3rd Place, InnovateX 2.0"
    assert p[1].tech == ["Flask", "HuggingFace Transformers (DistilBART, BART MNLI)", "Web Speech API"]


def test_certificate_detail_lines_stay_with_their_certificate():
    from resume_parser.rule_extractors import extract_certifications
    c = extract_certifications("Programming in Java | NPTEL | Oct 2024\nScore: 66% | ID: X1\nMeta React Native | Coursera | Jul 2024")
    assert len(c) == 2 and "Score: 66%" in c[0]


def test_location_from_contact_line_and_new_sections():
    from resume_parser.rule_extractors import extract_location
    from resume_parser.segmentation import segment
    assert extract_location(["Ana Roy", "+91 8160562120 | ana@x.com | Bharuch, Gujarat, India"], "Ana Roy") == "Bharuch, Gujarat, India"
    secs = segment("Ana Roy\nProjects\nP1 | Python\nAchievements & Awards\n- Won 1st place\nPatent Publication\n- Filed a patent\nSkills\nPython")
    assert {"projects", "achievements", "patents", "skills"} <= set(secs) and "Won 1st place" not in secs["projects"]


# ------------------------------------------------------------------ achievements / patents / languages / education range
def test_education_keeps_start_and_end_year():
    from resume_parser.rule_extractors import extract_education
    e = extract_education("Manipal University Jaipur\nBachelor of Technology in Data Science Aug 2023 – May 2027\nClass XII – CBSE | 85.0% 2022 – 2023")
    assert (e[0].start_year, e[0].year) == ("2023", "2027")
    assert (e[1].start_year, e[1].year) == ("2022", "2023")


def test_single_year_education_has_no_start_year():
    from resume_parser.rule_extractors import extract_education
    e = extract_education("XYZ College\nB.Sc Physics 2023")
    assert e[0].start_year is None and e[0].year == "2023"


def test_achievements_keep_titles_not_wrapped_bullet_text():
    from resume_parser.rule_extractors import extract_achievements
    got = extract_achievements("3rd Place – InnovateX 2.0 Nov 2025\n- Won 3rd place among\nteams at the university.\n⋆ Code for Good Selection – J.P. Morgan 2025\n- Selected nationally.")
    assert got == ["3rd Place – InnovateX 2.0", "Code for Good Selection – J.P. Morgan"]


def test_patent_fields():
    from resume_parser.rule_extractors import extract_patents
    p = extract_patents("Verifiable Mental Health System Filed: Dec 2025\nAdvanced Therapy (Shortlisted) Published: Feb 2026\n"
                        "Application No. 202511133248 A | Journal No. 06/2026\n- Invented a platform that\nwraps onto a second line.\n- Status: Application Awaiting Examination.")[0]
    assert p.title == "Verifiable Mental Health System Advanced Therapy (Shortlisted)"
    assert (p.application_number, p.filed, p.published) == ("202511133248", "Dec 2025", "Feb 2026")
    assert p.status == "Application Awaiting Examination"


def test_languages_with_and_without_levels():
    from resume_parser.rule_extractors import extract_languages
    assert extract_languages("English (Fluent) Hindi (Fluent) Gujarati (Intermediate)") == ["English (Fluent)", "Hindi (Fluent)", "Gujarati (Intermediate)"]
    assert extract_languages("English, Hindi | Gujarati") == ["English", "Hindi", "Gujarati"]


def test_languages_heading_is_its_own_section_and_skill_line_is_not_a_heading():
    from resume_parser.segmentation import segment
    s = segment("Technical Skills\nLanguages: Python, C\nLanguages\nEnglish (Fluent)")
    assert s["skills"] == "Languages: Python, C" and s["languages"] == "English (Fluent)"


def test_merge_carries_new_fields_from_llm_and_masks_start_year():
    from resume_parser.pipeline import merge
    from resume_parser.privacy import anonymize
    from resume_parser.schema import Education, ParsedResume
    from resume_parser.taxonomy import SkillTaxonomy
    llm = ParsedResume.model_validate({"achievements": ["3rd Place - InnovateX"], "languages": ["Hindi (Fluent)"],
                                       "patents": [{"title": "T", "application_number": "1"}],
                                       "education": [{"degree": "BTech", "start_year": "2023", "year": "2027"}]})
    out = merge(ParsedResume(), llm, SkillTaxonomy.from_file("data/skills.json"))
    assert out.achievements == ["3rd Place - InnovateX"] and out.languages == ["Hindi (Fluent)"] and out.patents[0].title == "T"
    assert out.sources["patents"] == "llm"
    ed = anonymize(out).education[0]
    assert ed.start_year is None and ed.year is None