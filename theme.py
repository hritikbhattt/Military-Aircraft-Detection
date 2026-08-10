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
        header { display: none !important; visibility: hidden !important; height: 0 !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        button[title="Open sidebar"] { display: none !important; }
        button[title="Close sidebar"] { display: none !important; }
        .stApp { background-color: #FFFFFF; }
        .block-container { padding-top: 0.6rem; padding-left: 2rem; padding-right: 2rem; max-width: 1180px; }

        .hero-wrap {
            background-color: #FAFAFA;
            background-image:
                linear-gradient(#EDEFF2 1px, transparent 1px),
                linear-gradient(90deg, #EDEFF2 1px, transparent 1px);
            background-size: 26px 26px;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.4rem;
        }
        .section-divider {
            height: 1px; background: #E5E7EB; margin: 1.6rem 0; border: none;
        }
        .bar-row { margin: 10px 0 16px 0; }
        .bar-label {
            display: flex; justify-content: space-between; font-size: 0.85rem;
            color: #4B5563; font-weight: 600; margin-bottom: 5px;
        }
        .bar-track {
            background: #E5E7EB; border-radius: 8px; height: 10px; overflow: hidden;
        }
        .bar-fill {
            height: 100%; border-radius: 8px;
            background: linear-gradient(90deg, #FDBA74, #EA580C);
            width: 0%;
            animation: growBar 1.1s ease-out forwards;
        }
        @keyframes growBar { to { width: var(--target-width); } }

        .navbar {
            position: sticky; top: 0; z-index: 999;
            display: flex; align-items: center; justify-content: space-between;
            background: rgba(255,255,255,0.92); backdrop-filter: blur(8px);
            border-bottom: 1px solid #F0E4DA;
            width: 100%; padding: 0.65rem 0.3rem; margin: 0 0 1.5rem 0;
        }
        .brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
        .brand-badge {
            width: 38px; height: 38px; border-radius: 11px;
            background: linear-gradient(135deg, #FDBA74, #EA580C 60%, #C2410C);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 3px 10px rgba(234,88,12,0.35);
            transition: transform 0.2s ease;
        }
        .brand:hover .brand-badge { transform: scale(1.06) rotate(-4deg); }
        .brand-name { font-weight: 800; color: #1F2937; font-size: 1.02rem; letter-spacing: -0.01em; }
        .nav-links { display: flex; gap: 4px; }
        .nav-link {
            position: relative;
            text-decoration: none; color: #6B7280; font-size: 0.9rem; font-weight: 600;
            padding: 8px 14px; border-radius: 8px; transition: all 0.18s ease;
        }
        .nav-link:hover { background: #FFF7ED; color: #EA580C; }
        .nav-link.active { background: linear-gradient(135deg, #FFF1E3, #FFE4CC); color: #C2410C; }

        .icon-badge {
            width: 42px; height: 42px; border-radius: 12px;
            background: linear-gradient(135deg, #FDBA74, #EA580C 60%, #C2410C);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 3px 10px rgba(234,88,12,0.3); flex-shrink: 0;
            transition: transform 0.2s ease;
        }
        .page-header:hover .icon-badge { transform: scale(1.05) rotate(-3deg); }
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
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .panel:hover { box-shadow: 0 6px 18px rgba(234,88,12,0.1); transform: translateY(-1px); }
        .desc-card { color: #4B5563; line-height: 1.55; }
        .panel h3, .panel h4 { margin-top: 0; color: #1F2937; }
        .panel p, .panel li { color: #4B5563; line-height: 1.6; }

        .stat-row { display: flex; gap: 14px; margin-bottom: 1.4rem; flex-wrap: wrap; }
        .stat-card {
            flex: 1; min-width: 160px; background: #FFFFFF; border: 1px solid #F0E4DA;
            border-radius: 14px; padding: 1.05rem 1.2rem; box-shadow: 0 1px 4px rgba(234,88,12,0.05);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .stat-card:hover { box-shadow: 0 8px 20px rgba(234,88,12,0.14); transform: translateY(-2px); }
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
            transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
        }
        .contact-card:hover {
            border-color: #EA580C; box-shadow: 0 8px 20px rgba(234,88,12,0.14);
            transform: translateY(-2px);
        }
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
