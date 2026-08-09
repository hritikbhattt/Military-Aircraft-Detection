import streamlit as st

LOGO_ICON = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="10" cy="10" r="6.2" stroke="white" stroke-width="2"/>
<line x1="14.3" y1="14.3" x2="20" y2="20" stroke="white" stroke-width="2" stroke-linecap="round"/>
<path d="M6.8 10L13.2 6.8L10.4 10L13.2 13.2L6.8 10Z" fill="white"/>
</svg>"""

INFO_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="12" r="9" stroke="white" stroke-width="2"/>
<circle cx="12" cy="8" r="1.1" fill="white"/>
<line x1="12" y1="11" x2="12" y2="17" stroke="white" stroke-width="2" stroke-linecap="round"/>
</svg>"""

CHART_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<line x1="4" y1="20" x2="20" y2="20" stroke="white" stroke-width="2" stroke-linecap="round"/>
<rect x="6" y="13" width="3" height="7" rx="0.8" fill="white"/>
<rect x="11" y="8" width="3" height="12" rx="0.8" fill="white"/>
<rect x="16" y="4" width="3" height="16" rx="0.8" fill="white"/>
</svg>"""

MAIL_ICON = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="3.5" y="5.5" width="17" height="13" rx="2" stroke="white" stroke-width="2"/>
<path d="M4.5 7L12 12.5L19.5 7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

PAGES = [
    {"label": "Home", "path": "streamlit_app.py", "url": "/"},
    {"label": "About", "path": "pages/1_About.py", "url": "/About"},
    {"label": "Results", "path": "pages/2_Results.py", "url": "/Results"},
    {"label": "Contact", "path": "pages/3_Contact.py", "url": "/Contact"},
]


def inject_css():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        .stApp { background-color: #FFFFFF; }
        .block-container { padding-top: 1rem; max-width: 1100px; }

        .navbar {
            position: sticky; top: 0; z-index: 999;
            display: flex; align-items: center; justify-content: space-between;
            background: #FFFFFF; border-bottom: 1px solid #F0E4DA;
            padding: 0.7rem 0.2rem; margin: -1rem -1rem 1.6rem -1rem;
            padding-left: 1rem; padding-right: 1rem;
        }
        .brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
        .brand-badge {
            width: 36px; height: 36px; border-radius: 10px;
            background: linear-gradient(135deg, #FB923C, #EA580C);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 2px 6px rgba(234,88,12,0.25);
        }
        .brand-name { font-weight: 700; color: #1F2937; font-size: 1.02rem; }
        .nav-links { display: flex; gap: 4px; }
        .nav-link {
            text-decoration: none; color: #6B7280; font-size: 0.9rem; font-weight: 600;
            padding: 8px 14px; border-radius: 8px; transition: all 0.15s ease;
        }
        .nav-link:hover { background: #FFF7ED; color: #EA580C; }
        .nav-link.active { background: #FFEDD5; color: #EA580C; }

        .icon-badge {
            width: 42px; height: 42px; border-radius: 12px;
            background: linear-gradient(135deg, #FB923C, #EA580C);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 2px 8px rgba(234,88,12,0.22); flex-shrink: 0;
        }
        .page-header {
            display: flex; align-items: center; gap: 14px;
            padding: 1rem 1.3rem; background: #FFFFFF; border: 1px solid #F0E4DA;
            border-radius: 14px; margin-bottom: 1.3rem;
            box-shadow: 0 1px 4px rgba(234,88,12,0.06);
        }
        .page-header-title { font-size: 1.15rem; font-weight: 700; color: #1F2937; margin: 0; }
        .page-header-sub { font-size: 0.82rem; color: #9CA3AF; margin: 0; }

        .desc-card, .panel {
            background: #FFFFFF; border: 1px solid #F0E4DA; border-radius: 14px;
            padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;
            box-shadow: 0 1px 4px rgba(234,88,12,0.05);
        }
        .desc-card { color: #4B5563; line-height: 1.55; }
        .panel h3, .panel h4 { margin-top: 0; color: #1F2937; }
        .panel p, .panel li { color: #4B5563; line-height: 1.6; }

        .stat-row { display: flex; gap: 14px; margin-bottom: 1.4rem; flex-wrap: wrap; }
        .stat-card {
            flex: 1; min-width: 160px; background: #FFFFFF; border: 1px solid #F0E4DA;
            border-radius: 14px; padding: 1.05rem 1.2rem; box-shadow: 0 1px 4px rgba(234,88,12,0.05);
        }
        .stat-card .lbl {
            font-size: 0.76rem; color: #9CA3AF; text-transform: uppercase;
            letter-spacing: 0.03em; font-weight: 600; margin-bottom: 4px;
        }
        .stat-card .num { font-size: 1.85rem; font-weight: 700; color: #1F2937; }
        .stat-card .accent { color: #EA580C; }

        .detection-chip {
            display: inline-flex; align-items: center; gap: 6px; background: #FFF7ED;
            border: 1px solid #FED7AA; border-radius: 8px; padding: 6px 12px;
            margin: 4px 6px 4px 0; font-size: 0.88rem; color: #C2410C;
        }
        .detection-chip b { color: #C2410C; }

        .tag {
            display: inline-flex; background: #FFF7ED; border: 1px solid #FED7AA;
            border-radius: 8px; padding: 5px 12px; margin: 3px 5px 3px 0;
            font-size: 0.85rem; color: #C2410C; font-weight: 500;
        }
        .contact-grid { display: flex; gap: 14px; flex-wrap: wrap; }
        .contact-card {
            flex: 1; min-width: 220px; background: #FFFFFF; border: 1px solid #F0E4DA;
            border-radius: 14px; padding: 1.3rem; box-shadow: 0 1px 4px rgba(234,88,12,0.05);
            text-decoration: none; display: block;
        }
        .contact-card:hover { border-color: #EA580C; }
        .contact-icon { margin-bottom: 8px; }
        .contact-label { font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase;
            letter-spacing: 0.03em; font-weight: 600; }
        .contact-value { font-size: 1rem; color: #1F2937; font-weight: 600; margin-top: 2px; }

        .badge-warn {
            background: #FFF7ED; color: #C2410C; border: 1px solid #FED7AA;
            padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        }
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def top_nav(active):
    links = ""
    for p in PAGES:
        cls = "nav-link active" if p["label"] == active else "nav-link"
        links += f'<a class="{cls}" href="{p["url"]}" target="_self">{p["label"]}</a>'
    st.markdown(f"""
    <div class="navbar">
        <a class="brand" href="/" target="_self">
            <div class="brand-badge">{LOGO_ICON}</div>
            <span class="brand-name">Military Aircraft Detection</span>
        </a>
        <div class="nav-links">{links}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(icon_svg, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="icon-badge">{icon_svg}</div>
        <div>
            <p class="page-header-title">{title}</p>
            <p class="page-header-sub">{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
