# app.py
import streamlit as st
import pandas as pd
import altair as alt
from groq import Groq
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException



from engine import (
    PortfolioConfig,
    load_all_data,
    run_backtest,
    run_today_optimization,
)

from functions import (validate_constraints,
                       compute_backtest_stats,
                       management_fee_from_wealth,
                       build_backtest_context_text,
                       build_constraints_summary,
                       format_constraints_block,
                       img_to_base64)


# --------------- GLOBAL DATA (cached) ---------------
@st.cache_data
def get_data():
    return load_all_data()


def main():
    st.set_page_config(
        page_title="QARM Portfolio Manager",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ("About us", "Our investment approach", "Phi assistant", "Portfolio optimization",
         "Become a Client", "Meet the Team"),
    )

    data = get_data()

    if page == "About us":
        page_about()
    elif page == "Portfolio optimization":
        page_portfolio_optimization(data)
    elif page == "Phi assistant":
        page_ai_assistant()
    elif page == "Our investment approach":
        page_investment_approach()
    elif page == "Become a Client":
        page_new_client()
    elif page == "Meet the Team":
        page_our_team()



def get_llm_client():
    """
    Returns a Groq client using the secret API key.
    """
    api_key = st.secrets["groq"]["api_key"]
    client = Groq(api_key=api_key)
    return client


# --------------- PAGE 1: ABOUT US ---------------
def page_about():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Open Sans', sans-serif;
            }

            /* HERO */
            .hero {
                background: linear-gradient(120deg, #133c55 0%, #1d4e6e 55%, #2a628f 100%);
                height: 240px;
                border-radius: 16px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                padding: 32px 46px;
                margin-bottom: 50px;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            }

            .hero-title {
                color: #ffffff;
                font-size: 44px;
                font-weight: 700;
                margin: 0;
                text-shadow: 0 2px 10px rgba(0,0,0,0.35);
            }

            .hero-subtitle {
                margin-top: 10px;
                color: #d9dde7;
                font-size: 18px;
                max-width: 1200px;
                line-height: 1.6;
            }

            /* SECTION TYPOGRAPHY */
            .section-heading {
                font-size: 30px;
                font-weight: 600;
                margin-bottom: 8px;
                color: #101827;
                margin-top: 36px;
            }

            .paragraph {
                font-size: 18px;
                color: #111827;
                line-height: 1.9;
                margin-bottom: 30px;
                max-width: 880px;
            }

            .paragraph strong {
                font-weight: 600;
            }

            /* IMAGE CARD */
            .image-frame {
                padding: 18px;
                border-radius: 22px;
                overflow: hidden;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20);
                background: radial-gradient(circle at top left, #f7f9ff 0%, #eef1f5 35%, #e2e5ec 100%);
                margin-top: 30px;
            }

            .image-frame img {
                width: 100%;
                height: 320px;
                object-fit: cover;
                border-radius: 22px;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20);
            }

            /* PILLARS ROW */
            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 18px;
                margin-top: 10px;
                margin-bottom: 10px;
            }

            .pill-card {
                flex: 1 1 180px;
                background: #f3f4f6;
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: 0 4px 10px rgba(15,23,42,0.06);
            }

            .pill-title {
                font-size: 15px;
                font-weight: 600;
                margin-bottom: 4px;
                color: #111827;
            }

            .pill-body {
                font-size: 14px;
                color: #4b5563;
                line-height: 1.6;
            }

            /* CONTACT BOX */
            .contact-box {
                padding: 28px 30px;
                background: #e8eef7;
                border-radius: 16px;
                box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
                max-width: 880px;
                margin-top: 70px;
            }

            .contact-box h3 {
                font-size: 24px;
                margin-bottom: 12px;
                color: #111827;
            }

            .contact-box p {
                font-size: 18px;
                margin: 6px 0;
                color: #374151;
            }

            .emoji {
                margin-right: 8px;
                font-size: 18px;
            }

            .tagline {
                font-style: italic;
                color: #4b5563;
                font-size: 16px;
                text-align: center;
                margin-top: 55px;
            }

            .bullet-list {
                font-size: 18px;
                color: #111827;
                line-height: 1.9;
                max-width: 880px;
                margin-bottom: 30px;
                padding-left: 18px;
            }

            .bullet-list li {
                margin-bottom: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # HERO
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Phi Investment Capital</div>
            <div class="hero-subtitle">
                A quantitative portfolio design studio that turns institutional research into
                transparent, explainable portfolios for long-term investors.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # WHO WE ARE
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<div id="who" class="section-heading">Who We Are</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="paragraph">
            Phi Investment Capital is a quantitative investment studio focused on building disciplined, 
            transparent, and evidence-based portfolios. Our team brings together experience in asset 
            management, quantitative research, and risk engineering, with a shared belief that investors 
            deserve tools that are powerful <em>and</em> understandable.
            </div>
            <div class="paragraph">
            We sit at the intersection of financial theory, data science, and practical portfolio 
            implementation. Our role is not to replace the allocator, but to give them a precise, 
            rule-based engine to express their views, control risk, and communicate clearly with their 
            own stakeholders.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image("invest_future.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr />', unsafe_allow_html=True)

    # WHAT WE DO
    st.markdown('<div id="what" class="section-heading">What We Do</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="paragraph">
        <strong>Phi is a portfolio design engine for sophisticated investors.</strong><br>
        Starting from institutional-grade universes such as the S&amp;P 500 and MSCI World, our platform 
        allows clients to encode their investment beliefs, risk appetite, and ESG or sector preferences 
        directly into the portfolio construction process. 
        </div>
        <div class="paragraph">
        Behind the interface, we use long-only mean–variance optimisation, robust covariance estimation, 
        rolling windows, and disciplined rebalancing. On the surface, clients see what matters most: 
        clear trade-offs, intuitive analytics, and portfolios that can be explained to an investment 
        committee in plain language.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr />', unsafe_allow_html=True)

    # WHY PHI
    st.markdown('<div id="why" class="section-heading">Why Phi</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="paragraph">
        Most portfolio tools fall into one of two extremes: simple but rigid templates designed for 
        retail investors, or powerful quantitative engines that require a dedicated quant team to use. 
        Phi is deliberately built for the space in between. It is <strong>institutional in depth</strong>, 
        yet <strong>practical and transparent in use</strong>.
        </div>
        <div class="paragraph">
        Every setting from estimation windows and rebalancing rules to ESG filters, sector limits, and 
        asset-class constraints is visible and documented. No hidden knobs, no opaque overrides. 
        Phi acts as a quantitative co-pilot, providing structure and discipline while keeping the allocator 
        firmly in control.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # HOW WE OPERATE – three small pillars
    st.markdown(
        """
        <div class="pill-row">
            <div class="pill-card">
                <div class="pill-title">Transparency</div>
                <div class="pill-body">
                    Every assumption is explicit: data windows, costs, constraints, and risk settings 
                    are all visible and reproducible.
                </div>
            </div>
            <div class="pill-card">
                <div class="pill-title">Discipline</div>
                <div class="pill-body">
                    Portfolios follow pre-defined rules, reducing emotional decisions and ensuring 
                    consistency across market environments.
                </div>
            </div>
            <div class="pill-card">
                <div class="pill-title">Clarity</div>
                <div class="pill-body">
                    Complex quantitative work is translated into simple explanations, charts, and 
                    narratives that clients can share.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr />', unsafe_allow_html=True)

    # WHO WE SERVE
    col6, col7 = st.columns([1, 1.5])
    with col6:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image("Team.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col7:
        st.markdown('<div class="section-heading">Who We Serve</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="bullet-list">
                <li>Family offices looking to formalise and document their investment process</li>
                <li>Wealth managers and private banks seeking transparent, rules-based portfolios</li>
                <li>Institutional allocators and investment committees needing robust scenario analysis</li>
                <li>Independent advisors who must clearly explain “why this portfolio” to their clients</li>
            </ul>
            <div class="paragraph">
            What our clients share is a focus on transparency, auditability, and the ability to connect 
            quantitative rigour with real-world conversations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr />', unsafe_allow_html=True)

    # OUR PHILOSOPHY
    st.markdown('<div id="philosophy" class="section-heading">Our Philosophy</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="paragraph">
        We believe that good investing is less about prediction and more about discipline. Every 
        allocation should have a clear rationale, every risk should be visible, and the behaviour of 
        a portfolio should be understood before capital is put to work.
        </div>
        <div class="paragraph">
        Phi was built to make that discipline tangible. By turning rigorous quantitative methods into a 
        controllable workflow, we help investors move from “I have a view” to “here is exactly how this 
        portfolio is constructed, how it behaves, and why it fits our objectives”.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr />', unsafe_allow_html=True)

    # CONTACT
    st.markdown('<div id="contact" class="section-heading"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="contact-box">
            <h3>Contact Us</h3>
            <p><span class="emoji">📧</span><strong>Email:</strong> phinvestments@hotmail.com</p>
            <p><span class="emoji">📍</span><strong>Address:</strong> 123 Financial Street, Geneva, Switzerland</p>
            <p><span class="emoji">📞</span><strong>Phone:</strong> +41 22 123 45 67</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='tagline'>Built with integrity. Driven by data. Designed for clarity.</div>",
        unsafe_allow_html=True,
    )



# --------------- PAGE 2: PORTFOLIO OPTIMIZATION ---------------
def page_portfolio_optimization(data):
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
      font-family: 'Open Sans', sans-serif;
    }

    /* HERO */
    .hero-opt {
      background: linear-gradient(120deg, #133c55 0%, #1d4e6e 55%, #2a628f 100%);
      min-height: 190px;
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 28px 40px;
      margin-bottom: 36px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }

    .hero-title-opt {
      color: #ffffff; 
      font-size: 36px;
      font-weight: 700;
      margin: 0;
    }

    .hero-subtitle-opt {
      margin-top: 10px;
      color: #d9dde7;
      font-size: 17px;
      max-width: 1200px;
      line-height: 1.6;
    }

    /* PARAGRAPH */
    .paragraph {
      font-size: 18px;
      color: #111827;
      line-height: 1.9;
      margin-bottom: 20px;
      max-width: 900px;
    }

    .paragraph strong {
      font-weight: 600;
    }

    /* STEP GRID */
    .step-grid-opt {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 16px;
      margin-top: 10px;
      margin-bottom: 28px;
    }

    .step-card-opt {
      background: #f9fafb;
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
    }

    .step-badge-opt {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      border-radius: 999px;
      background: #133c55;
      color: #ffffff;
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 6px;
    }

    .step-title-opt {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
      color: #111827;
    }

    .step-body-opt {
      font-size: 12px;
      color: #4b5563;
      line-height: 1.7;
    }
    </style>

    <div class="hero-opt">
      <div class="hero-title-opt">Portfolio Optimization</div>
      <div class="hero-subtitle-opt">
        Design and backtest a fully transparent, constrained multi-asset portfolio that reflects your universe,
        risk profile, ESG choices, and implementation preferences.
      </div>
    </div>

    <div class="step-grid-opt">

      <div class="step-card-opt">
        <div class="step-badge-opt">01</div>
        <div class="step-title-opt">Choose your universe & settings</div>
        <div class="step-body-opt">
          Select between S&amp;P 500, MSCI World, or a multi-asset setup and define key technical choices such as
          investment horizon, estimation window, and rebalancing frequency.
        </div>
      </div>

      <div class="step-card-opt">
        <div class="step-badge-opt">02</div>
        <div class="step-title-opt">Refine the investment universe</div>
        <div class="step-body-opt">
          Apply sector, ESG, and asset-class filters to align the investable set with your mandate, preferences,
          or exclusions before any optimisation takes place.
        </div>
      </div>

      <div class="step-card-opt">
        <div class="step-badge-opt">03</div>
        <div class="step-title-opt">Define your risk profile</div>
        <div class="step-body-opt">
          Answer a short risk profile questionnaire that translates your comfort with volatility and drawdowns
          into a quantitative risk-aversion parameter for the optimiser.
        </div>
      </div>

      <div class="step-card-opt">
        <div class="step-badge-opt">04</div>
        <div class="step-title-opt">Set portfolio constraints</div>
        <div class="step-body-opt">
          Impose limits on sectors, ESG buckets, asset classes, and per-asset weights. Optionally control turnover
          to manage implementation costs and trading intensity.
        </div>
      </div>

      <div class="step-card-opt">
        <div class="step-badge-opt">05</div>
        <div class="step-title-opt">Run backtest & current allocation</div>
        <div class="step-body-opt">
          Compute today’s optimal portfolio and a historical backtest. Review performance, risk, drawdowns, turnover,
          and fee impact, all under your chosen assumptions.
        </div>
      </div>

    </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # STEP 1 – GENERAL SETTINGS
    # ============================================================
    st.markdown("### Step 1 – General Settings")

    # ---- 4 columns for main settings ----
    colA, colB, colC, colD = st.columns(4)

    with colA:
        universe_choice = st.radio(
            "Equity Universe",
            options=["SP500", "MSCI"],
            format_func=lambda x: "S&P 500" if x == "SP500" else "MSCI World",
        )

    with colB:
        investment_amount = st.number_input(
            "Investment Amount",
            min_value=1_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
            help="Portfolio simulations and backtests will be expressed in this monetary amount.",
        )
        mgmt_fee_annual = management_fee_from_wealth(investment_amount)
        st.caption(
            f"Estimated annual management fee: **{mgmt_fee_annual:.2%}** "
            "(applied pro rata on a monthly basis)."
        )

    with colC:
        investment_horizon_years = st.selectbox(
            "Investment Horizon",
            options=[1, 2, 3, 5, 7, 10],
            index=0,
            format_func=lambda x: f"{x} year" if x == 1 else f"{x} years",
        )

    with colD:
        rebalance_label = st.selectbox(
            "Rebalancing Frequency",
            options=["Yearly", "Quarterly", "Monthly"],
            index=0,
        )
        if rebalance_label == "Yearly":
            rebalancing = 12
        elif rebalance_label == "Quarterly":
            rebalancing = 3
        else:
            rebalancing = 1

    # ---- Full-width estimation window block (no column) ----
    st.subheader("Estimation Window")

    use_custom_est = st.checkbox(
        "Enable custom estimation window",
        value=False,
        help=(
            "By default, the model uses 12 months of historical data to estimate "
            "expected returns and risk. Enable only if you understand the impact "
            "on estimation error and model stability."
        ),
    )

    if not use_custom_est:
        est_months = 12
        st.info("Using default setting: **12-month (1-year) estimation window**.")
    else:
        est_months = st.selectbox(
            "Select estimation window (in months)",
            options=[6, 12, 24, 36, 60],
            index=1,  # 12 months as the suggested default
            format_func=lambda m: f"{m} months",
        )

        st.warning(
            "**Caution:** Changing the estimation window alters the balance between "
            "**statistical reliability** and **reactivity to market regimes**. "
            "Shorter windows make the model more sensitive to recent moves but also "
            "more exposed to noise and unstable covariance estimates. Longer windows "
            "smooth short-term noise but may overweight outdated market conditions."
        )

    st.markdown("---")

    # ============================================================
    # STEP 2 – UNIVERSE & FILTERS
    # ============================================================
    st.markdown("### Step 2 – Universe & Filters")

    # --- Get metadata for chosen equity universe and other assets ---
    if universe_choice == "SP500":
        metadata_equity = data["metadata"]["SP500"]
    else:
        metadata_equity = data["metadata"]["MSCI"]

    metadata_other = data["metadata"]["Other"]

    # ---------- 2.1 Equity filters: sectors & ESG ----------
    st.subheader("Equity Filters")

    sectors_available = (
        metadata_equity["SECTOR"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    col_sect, col_esg = st.columns(2)

    with col_sect:
        selected_sectors = st.multiselect(
            "Sectors to include in equity universe",
            options=sectors_available,
            default=sectors_available,
            help="If you select all sectors, no sector filter is applied.",
        )

        if len(selected_sectors) == len(sectors_available) or len(selected_sectors) == 0:
            keep_sectors = None
        else:
            keep_sectors = selected_sectors

    with col_esg:
        esg_options = ["L", "M", "H"]
        selected_esg = st.multiselect(
            "ESG categories to include",
            options=esg_options,
            default=esg_options,
            help="L = Low, M = Medium, H = High. Selecting all applies no ESG filter.",
        )

        if len(selected_esg) == len(esg_options) or len(selected_esg) == 0:
            keep_esg = None
        else:
            keep_esg = selected_esg

    # ---------- 2.2 Other asset classes & instruments ----------
    st.subheader("Other Asset Classes")

    asset_classes_all = (
        metadata_other["ASSET_CLASS"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_asset_classes_other = st.multiselect(
        "Asset classes to include in the universe (beyond equity)",
        options=asset_classes_all,
        default=asset_classes_all,
        help=(
            "These asset classes will be available to the optimizer. "
            "Constraints later control how much can be allocated to each."
        ),
    )

    keep_ids_by_class = {}

    for ac in selected_asset_classes_other:
        subset = metadata_other[metadata_other["ASSET_CLASS"] == ac]
        ids_in_class = subset.index.astype(str).tolist()

        # Build pretty labels: TICKER – NAME  (fallbacks if missing)
        label_map: dict[str, str] = {}
        for idx, row in subset.iterrows():
            key = str(idx)

            ticker = None
            name = None
            if "TICKER" in subset.columns and pd.notna(row.get("TICKER")):
                ticker = str(row["TICKER"]).strip()
            if "NAME" in subset.columns and pd.notna(row.get("NAME")):
                name = str(row["NAME"]).strip()

            if ticker and name:
                label_map[key] = f"{ticker} – {name}"
            elif name:
                label_map[key] = name
            elif ticker:
                label_map[key] = ticker
            else:
                label_map[key] = key  # fallback: just the ID

        st.markdown(f"**{ac} instruments to include**")
        selected_ids = st.multiselect(
            f"Select {ac} instruments (leave all selected to keep full class)",
            options=ids_in_class,
            default=ids_in_class,
            format_func=lambda x: label_map.get(str(x), str(x)),
        )

        # Only store a filter if the user actually deselected something
        if 0 < len(selected_ids) < len(ids_in_class):
            keep_ids_by_class[ac] = selected_ids

    keep_ids_by_class = keep_ids_by_class if keep_ids_by_class else None

    st.markdown("---")

    # ============================================================
    # STEP 3 – RISK PROFILE QUESTIONNAIRE → GAMMA
    # ============================================================
    st.markdown("### Step 3 – Risk Profile Questionnaire")

    st.caption(
        "Answer each question on a 1–5 scale. "
        "1 = very conservative, 5 = very aggressive."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        q1 = st.slider(
            "1. Reaction to a -20% loss in one year\n"
            "1 = sell everything, 5 = buy more",
            min_value=1, max_value=5, value=3,
        )

        q2 = st.slider(
            "2. Comfort with large fluctuations\n"
            "1 = not at all, 5 = very comfortable",
            min_value=1, max_value=5, value=3,
        )

        q3 = st.slider(
            "3. Return vs risk trade-off\n"
            "1 = stable low returns, 5 = max return even with large risk",
            min_value=1, max_value=5, value=3,
        )

        q4 = st.slider(
            "4. Investment horizon\n"
            "1 = < 1 year, 5 = > 10 years",
            min_value=1, max_value=5, value=3,
        )

        q5 = st.slider(
            "5. How do you view risk?\n"
            "1 = something to avoid, 5 = essential for higher returns",
            min_value=1, max_value=5, value=3,
        )

    with col_right:
        q6 = st.slider(
            "6. Stress during market crashes\n"
            "1 = extremely stressed, 5 = not stressed at all",
            min_value=1, max_value=5, value=3,
        )

        q7 = st.slider(
            "7. Stability of your income/finances\n"
            "1 = very unstable, 5 = very stable",
            min_value=1, max_value=5, value=3,
        )

        q8 = st.slider(
            "8. Experience with investing\n"
            "1 = not familiar, 5 = very experienced",
            min_value=1, max_value=5, value=3,
        )

        q9 = st.slider(
            "9. Reaction to a +20% gain in one year\n"
            "1 = sell to lock gains, 5 = add significantly more money",
            min_value=1, max_value=5, value=3,
        )

        q10 = st.slider(
            "10. Share of net worth in risky assets\n"
            "1 = < 10%, 5 = > 60%",
            min_value=1, max_value=5, value=3,
        )

    scores = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    S = sum(scores)
    gamma = 0.5 + 2.5 * (S - 10)  # internal only

    if S <= 20:
        profile_label = "Very Conservative"
        profile_text = (
            "You have a **very low tolerance for risk** and prefer capital preservation. "
            "The portfolio will be tilted towards safer, lower-volatility assets."
        )
    elif S <= 30:
        profile_label = "Conservative"
        profile_text = (
            "You are **cautious with risk**, but willing to accept some fluctuations. "
            "The portfolio will prioritize stability with a moderate growth component."
        )
    elif S <= 35:
        profile_label = "Balanced"
        profile_text = (
            "You have a **balanced attitude** towards risk and return. "
            "The portfolio will mix growth assets with stabilizing components."
        )
    elif S <= 42:
        profile_label = "Dynamic"
        profile_text = (
            "You are **comfortable with risk** and seek higher returns. "
            "The portfolio will have a strong allocation to growth and risky assets."
        )
    else:
        profile_label = "Aggressive"
        profile_text = (
            "You have a **high risk tolerance** and focus on return maximization. "
            "The portfolio will be heavily exposed to volatile, return-seeking assets."
        )

    st.markdown("")
    col_score, col_profile = st.columns(2)
    with col_score:
        st.metric("Total Risk Score (S)", f"{S} / 50")
    with col_profile:
        st.markdown(f"**Risk Profile:** {profile_label}")
        st.caption(profile_text)

    st.markdown("---")

    # ============================================================
    # STEP 4 – CONSTRAINTS
    # ============================================================
    st.markdown("### Step 4 – Constraints")

    st.caption(
        "All constraints are expressed as **fractions** (0.10 = 10%). "
        "Leave min = 0 and max = 1 to avoid imposing a constraint."
    )

    # ------------------------------------------------------------
    # 4.1.1 Max weight per asset (with safe default + warning)
    # ------------------------------------------------------------
    st.subheader("Maximum Weight per Asset")

    use_custom_max = st.checkbox(
        "Enable custom maximum weight per asset",
        value=False,
        help="By default, each asset is capped at 5%. Enable only if you understand concentration risk."
    )

    if not use_custom_max:
        max_weight_per_asset = 0.05
        st.info("Using default limit: **5% maximum per individual asset**.")
    else:
        max_weight_per_asset = st.slider(
            "Select maximum weight per asset",
            min_value=0.01,
            max_value=0.25,
            value=0.05,
            step=0.01,
            help="Higher caps increase concentration risk and may reduce diversification."
        )

        st.warning(
            "**Caution:** Increasing the maximum weight per asset may significantly raise your "
            "**idiosyncratic risk** and reduce the portfolio's **diversification benefits**. "
            "Large individual exposures can amplify the impact of adverse movements in a single "
            "security, especially during periods of market stress."
        )

    st.markdown("---")

    # 4.1.2
    st.subheader("Turnover Constraint (per rebalance)")

    use_turnover_constr = st.checkbox(
        "Enable maximum turnover per rebalance",
        value=False,
        help=(
            "Turnover represents the total fraction of the portfolio traded each rebalance."
            "The turnover limit applies only from the second rebalance onward, since the "
            "first one is the initial investment from cash."
        ),
    )

    if not use_turnover_constr:
        max_turnover_per_rebalance = None
        st.info(
            "**No explicit turnover cap** is imposed by default. "
        )
    else:
        max_turnover_per_rebalance = st.slider(
            "Maximum one-way turnover per rebalance",
            min_value=0.10,
            max_value=1.00,
            value=0.50,
            step=0.05,
            # format="%.0f%%",
            help=(
                "For example, 30% means that at each rebalance (except the initial "
                "funding from cash), the total one-way traded volume cannot exceed "
                "30% of the portfolio value."
            ),
        )
        # Convert from % to fraction
        max_turnover_per_rebalance = float(max_turnover_per_rebalance)

        st.warning(
            "A tight turnover cap can significantly reduce rebalancing activity and "
            "transaction costs, as well as tracking error, but may also limit the "
            "optimizer’s ability to adapt to new information and improve risk/return."
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.2 Sector constraints within equity (relative to equity)
    # ------------------------------------------------------------
    st.subheader("Equity Sector Constraints (relative to the equity exposure)")

    if keep_sectors is None:
        sectors_for_constraints = sectors_available
    else:
        sectors_for_constraints = keep_sectors

    sector_constraints = {}
    sector_min_budget = 0.0  # sum of mins so far, must stay <= 1

    for sec in sectors_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - sector_min_budget)

        with st.expander(f"{sec}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                sec_min = st.number_input(
                    f"Min share of {sec} in Equity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"sec_min_{sec}",
                )

            # update budget after this min
            sector_min_budget += sec_min

            with col_max:
                sec_max = st.number_input(
                    f"Max share of {sec} in Equity",
                    min_value=0.0,  # ensures min <= max
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"sec_max_{sec}",
                )

            # Professional-style warnings at boundaries
            eps = 1e-8
            if remaining_min_budget > 0 and abs(sec_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **{sec}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of sector minima above **100% of the equity slice** "
                    "and is therefore not admissible."
                )

            if sec_min > 0 and abs(sec_max - sec_min) < eps:
                st.info(
                    f"For **{sec}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer to rebalance within this sector."
                )

        cons = {}
        if sec_min > 0:
            cons["min"] = float(sec_min)
        if sec_max < 1.0:
            cons["max"] = float(sec_max)
        if cons:
            sector_constraints[sec] = cons

    if not sector_constraints:
        sector_constraints = None

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.3 ESG constraints within equity (relative to equity)
    # ------------------------------------------------------------
    st.subheader("Equity ESG Score Constraints (relative to the equity exposure)")

    esg_all_labels = ["L", "M", "H"]
    if keep_esg is None:
        esg_for_constraints = esg_all_labels
    else:
        esg_for_constraints = keep_esg

    esg_constraints = {}
    esg_min_budget = 0.0

    for label in esg_for_constraints:
        remaining_min_budget = max(0.0, 1.0 - esg_min_budget)

        with st.expander(f"ESG {label}", expanded=False):
            col_min, col_max = st.columns(2)

            with col_min:
                esg_min = st.number_input(
                    f"Min share of ESG {label} score in Equity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"esg_min_{label}",
                )

            esg_min_budget += esg_min

            with col_max:
                esg_max = st.number_input(
                    f"Max share ESG {label} score in Equity",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"esg_max_{label}",
                )

            eps = 1e-8
            if remaining_min_budget > 0 and abs(esg_min - remaining_min_budget) < eps:
                st.warning(
                    f"The minimum allocation entered for **ESG {label}** is at the upper feasible bound. "
                    "Any higher minimum would force the sum of ESG minima above **100% of the equity slice** "
                    "and is therefore not admissible."
                )

            if esg_min > 0 and abs(esg_max - esg_min) < eps:
                st.info(
                    f"For **ESG {label}**, the minimum and maximum allocations are effectively identical. "
                    "This leaves no flexibility for the optimizer within this ESG bucket."
                )

        cons = {}
        if esg_min > 0:
            cons["min"] = float(esg_min)
        if esg_max < 1.0:
            cons["max"] = float(esg_max)
        if cons:
            esg_constraints[label] = cons

    if not esg_constraints:
        esg_constraints = None

    st.markdown("---")

    # ------------------------------------------------------------
    # 4.4 Asset-class constraints (total portfolio)
    # ------------------------------------------------------------
    st.subheader("Asset-Class Constraints (total portfolio)")

    if not selected_asset_classes_other:
        st.info(
            "You have selected an **equity-only universe**. "
            "By construction, 100% of the portfolio will be invested in Equity."
        )
        asset_class_constraints = None

    else:

        asset_classes_for_constraints = ["Equity"] + selected_asset_classes_other

        asset_class_constraints = {}
        ac_min_budget = 0.0

        for ac in asset_classes_for_constraints:
            remaining_min_budget = max(0.0, 1.0 - ac_min_budget)

            with st.expander(f"{ac}", expanded=False):
                col_min, col_max = st.columns(2)

                with col_min:
                    ac_min = st.number_input(
                        f"Min portfolio weight in {ac}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.2f",
                        key=f"ac_min_{ac}",
                    )

                ac_min_budget += ac_min

                with col_max:
                    ac_max = st.number_input(
                        f"Max portfolio weight in {ac}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        step=0.05,
                        format="%.2f",
                        key=f"ac_max_{ac}",
                    )

                eps = 1e-8
                if remaining_min_budget > 0 and abs(ac_min - remaining_min_budget) < eps:
                    st.warning(
                        f"The minimum allocation entered for **{ac}** is at the upper feasible bound. "
                        "Any higher minimum would force the sum of asset-class minima above **100% of the portfolio** "
                        "and is therefore not admissible."
                    )

                if ac_min > 0 and abs(ac_max - ac_min) < eps:
                    st.info(
                        f"For **{ac}**, the minimum and maximum allocations are effectively identical. "
                        "This leaves no flexibility for the optimizer to reallocate across asset classes."
                    )

            cons = {}
            if ac_min > 0:
                cons["min"] = float(ac_min)
            if ac_max < 1.0:
                cons["max"] = float(ac_max)
            if cons:
                asset_class_constraints[ac] = cons

        if not asset_class_constraints:
            asset_class_constraints = None

        st.markdown("---")

    constraint_errors = validate_constraints(
        sector_constraints=sector_constraints,
        esg_constraints=esg_constraints,
        asset_class_constraints=asset_class_constraints,
    )

    if constraint_errors:
        st.error("The current constraint configuration is not feasible:")
        for msg in constraint_errors:
            st.write(f"• {msg}")

    # ============================================================
    # STEP 5 – Backtest and Today's Portfolio (independent)
    # ============================================================
    st.markdown("### Step 5 – Backtest and Current Optimal Allocation")

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        run_backtest_clicked = st.button(
            "Run Backtest",
            type="primary",
            disabled=bool(constraint_errors),
            use_container_width=True,
        )

    with col_btn2:
        run_today_clicked = st.button(
            "Compute Current Optimal Allocation",
            type="primary",
            disabled=bool(constraint_errors),
            use_container_width=True,
        )

    # --- Helper: build config from current UI inputs ---
    def _build_config():
        return PortfolioConfig(
            today_date=pd.Timestamp("2025-10-01"),
            investment_horizon_years=investment_horizon_years,
            est_months=est_months,
            rebalancing=rebalancing,
            gamma=gamma,
            universe_choice=universe_choice,
            keep_sectors=keep_sectors,
            keep_esg=keep_esg,
            selected_asset_classes_other=selected_asset_classes_other,
            keep_ids_by_class=keep_ids_by_class,
            max_weight_per_asset=max_weight_per_asset,
            sector_constraints=sector_constraints,
            esg_constraints=esg_constraints,
            asset_class_constraints=asset_class_constraints,
            initial_wealth=investment_amount,
            max_turnover_per_rebalance=max_turnover_per_rebalance,
        )

    # --- 1) Run backtest only ---
    if run_backtest_clicked:
        if constraint_errors:
            st.error("The current constraint configuration is not feasible:")
            for msg in constraint_errors:
                st.write(f"• {msg}")
            st.stop()  # do not run the optimizer

        config = _build_config()

        try:
            with st.spinner("Running backtest..."):
                perf, summary_df, debug_weights_df = run_backtest(config, data)
        except ValueError:
            st.error(
                "The optimizer could not find a feasible portfolio with the current set of "
                "constraints and per-asset limits."
            )
            st.caption(
                "This typically happens when minimum allocations across sectors, ESG buckets or "
                "asset classes are too tight relative to the available universe and the maximum "
                "weight per asset. Please relax some minimum constraints or increase the maximum "
                "weight per asset, then try again."
            )
            st.stop()

        st.success("Backtest completed.")

        prev = st.session_state.get("backtest_results", {})
        st.session_state["backtest_results"] = {
            **prev,
            "perf": perf,
            "summary_df": summary_df,
            "debug_weights_df": debug_weights_df,
            "investment_amount": investment_amount,
            "universe_choice": universe_choice,
            "investment_horizon_years": investment_horizon_years,
            "est_months": est_months,
            "rebalancing": rebalancing,
            "gamma": gamma,
            "profile_label": profile_label,
            "max_weight_per_asset": max_weight_per_asset,
            "selected_asset_classes_other": selected_asset_classes_other,
            "sector_constraints": sector_constraints,
            "esg_constraints": esg_constraints,
            "asset_class_constraints": asset_class_constraints,
            "max_turnover_per_rebalance": max_turnover_per_rebalance,
            "keep_sectors": keep_sectors,
            "keep_esg": keep_esg,
        }

    # --- 2) Run today's optimal portfolio only ---
    if run_today_clicked:
        if constraint_errors:
            st.error("The current constraint configuration is not feasible:")
            for msg in constraint_errors:
                st.write(f"• {msg}")
            st.stop()

        config = _build_config()

        try:
            with st.spinner("Computing current optimal allocation..."):
                today_res = run_today_optimization(config, data)
        except ValueError:
            st.error(
                "The optimizer could not find a feasible current portfolio with the "
                "current universe and constraints."
            )
            st.stop()

        st.success("Current Optimal Allocation computed.")

        prev = st.session_state.get("backtest_results", {})
        st.session_state["backtest_results"] = {
            **prev,
            "today_res": today_res,
            "investment_amount": investment_amount,
            "universe_choice": universe_choice,
            "investment_horizon_years": investment_horizon_years,
            "est_months": est_months,
            "rebalancing": rebalancing,
            "gamma": gamma,
            "profile_label": profile_label,
            "max_weight_per_asset": max_weight_per_asset,
            "selected_asset_classes_other": selected_asset_classes_other,
            "sector_constraints": sector_constraints,
            "esg_constraints": esg_constraints,
            "asset_class_constraints": asset_class_constraints,
            "max_turnover_per_rebalance": max_turnover_per_rebalance,
            "keep_sectors": keep_sectors,
            "keep_esg": keep_esg,
        }

    # ============================================================
    # SHOW RESULTS TABS (BACKTEST / TODAY)
    # ============================================================
    if "backtest_results" in st.session_state:
        r = st.session_state["backtest_results"]

        perf = r.get("perf")
        summary_df = r.get("summary_df")
        debug_weights_df = r.get("debug_weights_df")
        today_res = r.get("today_res")

        investment_amount = r["investment_amount"]
        universe_choice = r["universe_choice"]
        investment_horizon_years = r["investment_horizon_years"]
        est_months = r["est_months"]
        rebalancing = r["rebalancing"]
        gamma = r["gamma"]
        profile_label = r["profile_label"]
        max_weight_per_asset = r["max_weight_per_asset"]
        selected_asset_classes_other = r["selected_asset_classes_other"]
        sector_constraints = r["sector_constraints"]
        esg_constraints = r["esg_constraints"]
        asset_class_constraints = r["asset_class_constraints"]

        tab_backtest, tab_today = st.tabs(["Backtest", "Current Optimal Allocation"])

        # ============================
        # TAB 1 – BACKTEST
        # ============================
        with tab_backtest:
            st.subheader("Backtest Performance")

            if perf is not None and not perf.empty:
                # 1) Compute backtest stats (for max drawdown, etc.)
                stats = compute_backtest_stats(perf)

                # --------------------------------------------------------
                # A) BUILD DATAFRAME WITH PORTFOLIO + BENCHMARKS (CUMRET)
                # --------------------------------------------------------
                returns_bench = data.get("benchmarks", None)

                if returns_bench is not None and not returns_bench.empty:
                    bench = returns_bench.reindex(perf.index)
                    bench_cum = (1.0 + bench).cumprod() - 1.0
                    combined = pd.concat([perf["CumReturn"], bench_cum], axis=1)
                    combined.columns = ["Portfolio"] + list(bench_cum.columns)
                else:
                    combined = pd.DataFrame({"Portfolio": perf["CumReturn"]})

                # Convert index to timestamp
                if isinstance(combined.index, pd.PeriodIndex):
                    combined.index = combined.index.to_timestamp()

                chart_data = combined.reset_index().rename(columns={"Date": "Date"})

                chart_data_long = chart_data.melt("Date", var_name="Series", value_name="Return")

                # --------------------------------------------------------
                # B) CUMULATIVE RETURN CHART WITH BENCHMARKS
                # --------------------------------------------------------
                st.markdown("**Cumulative Return of the Strategy vs Benchmarks**")

                max_ret = float(chart_data_long["Return"].max()) if not chart_data_long["Return"].isna().all() else 0.0
                max_ret = max(max_ret, 0.0)
                max_tick = (int(max_ret * 10) + 1) / 10.0 if max_ret > 0 else 0.1
                tick_values = [i / 10.0 for i in range(0, int(max_tick * 10) + 1)]

                base_ret = (
                    alt.Chart(chart_data_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y(
                            "Return:Q",
                            title="Cumulative return",
                            scale=alt.Scale(domain=[0, max_tick], nice=False),
                            axis=alt.Axis(format="%", values=tick_values),
                        ),
                        color=alt.Color("Series:N", sort=["Portfolio", "S&P 500", "MSCI WORLD"]),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date", format="%b %Y"),
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Return:Q", title="Cumulative return", format=".2%"),
                        ],
                    )
                )

                # Drawdown vertical lines
                if stats and stats["max_drawdown_start"] is not None:
                    dd_start, dd_end = stats["max_drawdown_start"], stats["max_drawdown_end"]
                    vline_data = pd.DataFrame({"Date": [dd_start, dd_end], "Label": ["DD start", "DD end"]})

                    vlines = (
                        alt.Chart(vline_data)
                        .mark_rule(color="red", strokeDash=[4, 4], size=2)
                        .encode(x="Date:T")
                    )

                    chart_ret = alt.layer(base_ret, vlines).interactive()
                else:
                    chart_ret = base_ret.interactive()

                st.altair_chart(chart_ret, use_container_width=True)

                st.markdown(
                    "<span style='color:red;'>Red dashed vertical lines indicate the "
                    "<b>start</b> and <b>end</b> of the worst drawdown observed over the backtest period.</span>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # --------------------------------------------------------
                # C) SECOND CHART – PORTFOLIO WEALTH
                # --------------------------------------------------------
                st.markdown("**Evolution of Portfolio Wealth**")

                perf_plot = perf.copy()
                if isinstance(perf_plot.index, pd.PeriodIndex):
                    perf_plot.index = perf_plot.index.to_timestamp()
                perf_plot = perf_plot.reset_index()

                max_wealth = float(perf["Wealth"].max())
                upper_wealth = max(max_wealth, 1.0) * 1.05

                base_wealth = (
                    alt.Chart(perf_plot)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
                        y=alt.Y("Wealth:Q", scale=alt.Scale(domain=[0, upper_wealth])),
                        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Wealth:Q", format=",.0f")],
                    )
                )

                if stats and stats["max_drawdown_start"] is not None:
                    chart_wealth = alt.layer(base_wealth, vlines).interactive()
                else:
                    chart_wealth = base_wealth.interactive()

                st.altair_chart(chart_wealth, use_container_width=True)

                # --------------------------------------------------------
                # D) TURNOVER PER REBALANCE (BAR CHART)
                # --------------------------------------------------------
                st.markdown("**Turnover per Rebalance**")

                if "Turnover" in perf.columns:
                    # Only months with a rebalance (Turnover > 0)
                    turnover_df = perf[perf["Turnover"] > 0][["Turnover"]].copy()

                    if not turnover_df.empty:
                        # Drop the initial funding trade from the chart
                        first_idx = turnover_df.index[0]
                        turnover_df = turnover_df.drop(first_idx)

                    if not turnover_df.empty:
                        # Convert Period → Timestamp for display
                        idx = turnover_df.index
                        if isinstance(idx, pd.PeriodIndex):
                            turnover_df.index = idx.to_timestamp()

                        turnover_df_reset = turnover_df.reset_index().rename(columns={"index": "Date"})
                        turnover_df_reset["DateLabel"] = turnover_df_reset["Date"].dt.strftime("%b %Y")

                        # Scale Y-axis nicely
                        max_turn = float(turnover_df_reset["Turnover"].max())
                        upper_turn = max_turn * 1.1 if max_turn > 0 else 0.1

                        chart_turnover = (
                            alt.Chart(turnover_df_reset)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "DateLabel:N",
                                    title="Rebalance date",
                                    sort=None,
                                    axis=alt.Axis(labelAngle=0),
                                ),
                                y=alt.Y(
                                    "Turnover:Q",
                                    title="One-way turnover",
                                    axis=alt.Axis(format=".0%"),
                                    scale=alt.Scale(domain=[0, upper_turn]),
                                ),
                                tooltip=[
                                    alt.Tooltip("Date:T", title="Date", format="%b %Y"),
                                    alt.Tooltip("Turnover:Q", title="Turnover", format=".2%"),
                                ],
                            )
                        ).properties(height=250)

                        st.altair_chart(chart_turnover, use_container_width=True)
                    else:
                        st.info("No rebalances with non-zero turnover (excluding initial funding trade).")
                else:
                    st.info("Turnover data not available.")

                # --------------------------------------------------------
                # E) BACKTEST STATISTICS TABLE
                # --------------------------------------------------------
                st.markdown("## Backtest Statistics")

                def show_stats_block(title, entries):
                    df_block = pd.DataFrame(entries, columns=["Metric", "Value"])
                    st.markdown(f"### {title}")
                    st.table(df_block)

                initial = investment_amount
                final_wealth = investment_amount * float(perf["Growth"].iloc[-1])

                def pct(x):
                    return f"{x:.2%}"

                total_tx_fees = float(perf["TxFeeAmount"].sum()) if "TxFeeAmount" in perf else 0
                total_mgmt_fees = float(perf["MgmtFeeAmount"].sum()) if "MgmtFeeAmount" in perf else 0

                wealth_block = [
                    ("Initial invested wealth", f"{initial:,.0f}"),
                    ("Final wealth at end of backtest", f"{final_wealth:,.0f}"),
                    ("Total management fees paid", f"{total_mgmt_fees:,.0f}"),
                    ("Total transaction costs paid", f"{total_tx_fees:,.0f}"),
                ]
                show_stats_block("Wealth & Fees Summary", wealth_block)

                performance_block = [
                    ("Annualised average return", pct(stats["annualised_avg_return"])),
                    ("Annualised volatility", pct(stats["annualised_volatility"])),
                    ("Annualised cumulative return", pct(stats["annualised_cum_return"])),
                    ("Min monthly return", pct(stats["min_monthly_return"])),
                    ("Max monthly return", pct(stats["max_monthly_return"])),
                ]
                show_stats_block("Performance Statistics", performance_block)

                drawdown_block = [
                    ("Max drawdown", pct(stats["max_drawdown"])),
                    ("Max drawdown start", stats["max_drawdown_start"].strftime("%b %Y")),
                    ("Max drawdown end", stats["max_drawdown_end"].strftime("%b %Y")),
                    ("Max drawdown duration", f"{stats['max_drawdown_duration_months']} months"),
                ]
                show_stats_block("Drawdown Analysis", drawdown_block)

                if "total_turnover" in stats:
                    turnover_block = [
                        ("Total one-way turnover", pct(stats["total_turnover"])),
                        ("Average turnover per rebalance", pct(stats["avg_turnover_per_rebalance"])),
                    ]
                    show_stats_block("Turnover & Trading Activity", turnover_block)

                # --------------------------------------------------------
                # F) AI Commentary on the Backtest
                # --------------------------------------------------------
                st.markdown("### AI Commentary on Backtest Results")

                explain_btn = st.button(
                    "Generate AI Commentary on Backtest",
                    type="secondary",
                    help="Ask the Phi Investment Capital digital assistant to provide a "
                         "client-friendly interpretation of the backtest results.",
                )

                if explain_btn:
                    client_llm = get_llm_client()

                    context_text = build_backtest_context_text(
                        stats=stats,
                        perf=perf,
                        investment_amount=investment_amount,
                        universe_choice=universe_choice,
                        investment_horizon_years=investment_horizon_years,
                        est_months=est_months,
                        rebalancing=rebalancing,
                        gamma=gamma,
                        profile_label=profile_label,
                        max_weight_per_asset=max_weight_per_asset,
                        selected_asset_classes_other=selected_asset_classes_other,
                        sector_constraints=sector_constraints,
                        esg_constraints=esg_constraints,
                        asset_class_constraints=asset_class_constraints,
                    )

                    system_prompt = (
                        "You are a digital investment assistant for Phi Investment Capital. "
                        "You are given a summary of a client's configuration and portfolio backtest on his configurations. "
                        "Provide a professional, client-friendly commentary on the results. And remember it's not our rela client"
                        "it's our potential client, he just go to platform and run optimization with his preferences and see what "
                        "would be the performace if he has invested with us for the horizon he choose. \n\n"
                        "Guidelines:\n"
                        "- Don't talk about turnover if the investment horizon is 1 year with yearly rebalance. It will be 0% anyway as no rebalancement happen"
                        "- Don't talk about gamma score just the risk profile from the score"
                        "- Refer to the client's risk profile, constraints, and investment horizon when relevant.\n"
                        "- Comment on constraints the client choose"
                        "- Comment on the balance between return and risk (volatility and drawdowns).\n"
                        "- Highlight any notable features of the drawdown profile and overall behaviour over time.\n"
                        "- You may mention that tighter constraints or lower max weights can limit performance but improve diversification.\n"
                        "- Do NOT give investment recommendations or instructions to buy/sell.\n"
                        "- Do NOT make promises about future performance.\n"
                        "- Keep the answer to about 2–5 short paragraphs, in a calm and professional tone."
                    )

                    user_message = (
                        "Here is the full context (client configuration and backtest summary). "
                        "Please provide a concise commentary for the client:\n\n"
                        f"{context_text}"
                    )

                    with st.spinner("Generating AI commentary..."):
                        response = client_llm.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message},
                            ],
                        )
                        commentary = response.choices[0].message.content

                    st.markdown(
                        """
                        **Phi Investment Capital – Backtest Commentary**  
                        *(Generated by the digital assistant based on your inputs and the statistics above.)*
                        """
                    )
                    st.markdown(commentary)
            else:
                st.info(
                    "No backtest results yet. Click **'Run Backtest'** to compute the historical performance."
                )

        # ============================
        # TAB 2 – TODAY'S PORTFOLIO
        # ============================
        with tab_today:
            st.subheader("Current Optimal Allocation")

            if today_res is None:
                st.info(
                    "No current portfolio yet. Click **'Compute Current Optimal Allocation'** "
                    "to generate the allocation."
                )
            else:
                # ---------- Unpack today's optimization results ----------
                today_df = today_res["weights"]
                top5 = today_res["top5"]
                alloc_by_ac = today_res["alloc_by_asset_class"]
                sector_in_eq = today_res["sector_in_equity"]
                esg_in_eq = today_res["esg_in_equity"]
                within_non_eq = today_res["within_non_equity_classes"]
                candidates_period = today_res["candidates_period"]

                # ---------- Header summary ----------
                as_of_date = (
                    (candidates_period + 1).to_timestamp().strftime("%b %Y")
                    if hasattr(candidates_period, "to_timestamp")
                    else str(candidates_period)
                )

                total_holdings = len(today_df)
                equity_weight = float(
                    today_df.loc[today_df["ASSET_CLASS"] == "Equity", "Weight"].sum()
                )
                n_asset_classes = today_df["ASSET_CLASS"].nunique()

                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("As of", as_of_date)
                with col_info2:
                    st.metric("Number of holdings", f"{total_holdings}")
                with col_info3:
                    st.metric("Equity slice", f"{equity_weight:.1%}")


                st.markdown("---")

                # ---------- Top 5 holdings ----------
                st.markdown("### Top 5 Holdings (by portfolio weight)")
                top5_display = top5.copy()
                top5_display["Weight"] = top5_display["Weight"].map(lambda x: f"{x:.2%}")
                st.dataframe(
                    top5_display[["TICKER", "NAME", "ASSET_CLASS", "SECTOR", "ESG", "Weight"]],
                    use_container_width=True,
                )

                st.markdown("---")

                # ---------- Asset class / Sector / ESG charts ----------
                colA, colB, colC = st.columns(3)

                # --- By Asset Class ---
                with colA:
                    st.markdown("**Allocation by Asset Class**")

                    if not alloc_by_ac.empty:
                        df_ac = alloc_by_ac.reset_index()
                        df_ac.columns = ["AssetClass", "Weight"]

                        df_ac["Min"] = df_ac["AssetClass"].map(
                            lambda ac: asset_class_constraints.get(ac, {}).get("min", None)
                            if asset_class_constraints
                            else None
                        )
                        df_ac["Max"] = df_ac["AssetClass"].map(
                            lambda ac: asset_class_constraints.get(ac, {}).get("max", None)
                            if asset_class_constraints
                            else None
                        )

                        df_ac["Min_end"] = df_ac["Min"] + 0.01
                        df_ac["Max_end"] = df_ac["Max"] + 0.01

                        bars_ac = (
                            alt.Chart(df_ac)
                            .mark_bar(color="#4BA3FF")
                            .encode(
                                x=alt.X("AssetClass:N", title=""),
                                y=alt.Y("Weight:Q", title="Portfolio weight"),
                                tooltip=[
                                    alt.Tooltip("AssetClass:N", title="Asset class"),
                                    alt.Tooltip("Weight:Q", title="Weight", format=".2%"),
                                ],
                            )
                        )

                        min_marks_ac = (
                            alt.Chart(df_ac)
                            .mark_bar(color="yellow", opacity=0.9)
                            .encode(
                                x=alt.X("AssetClass:N", title=""),
                                y=alt.Y("Min:Q"),
                                y2=alt.Y2("Min_end:Q"),
                                tooltip=[alt.Tooltip("Min:Q", title="Min", format=".2%")],
                            )
                            .transform_filter("datum.Min != null")
                        )

                        max_marks_ac = (
                            alt.Chart(df_ac)
                            .mark_bar(color="red", opacity=0.9)
                            .encode(
                                x=alt.X("AssetClass:N", title=""),
                                y=alt.Y("Max:Q"),
                                y2=alt.Y2("Max_end:Q"),
                                tooltip=[alt.Tooltip("Max:Q", title="Max", format=".2%")],
                            )
                            .transform_filter("datum.Max != null")
                        )

                        chart_ac = (bars_ac + min_marks_ac + max_marks_ac).properties(
                            height=300
                        ).interactive()

                        st.altair_chart(chart_ac, use_container_width=True)
                    else:
                        st.info("No allocation across asset classes.")

                # --- Sector (within Equity) ---
                with colB:
                    st.markdown("**Sector Breakdown (within Equity)**")

                    if not sector_in_eq.empty:
                        df_sector = sector_in_eq.reset_index()
                        df_sector.columns = ["Sector", "Weight"]

                        df_sector["Min"] = df_sector["Sector"].map(
                            lambda s: sector_constraints.get(s, {}).get("min", None)
                            if sector_constraints
                            else None
                        )
                        df_sector["Max"] = df_sector["Sector"].map(
                            lambda s: sector_constraints.get(s, {}).get("max", None)
                            if sector_constraints
                            else None
                        )

                        df_sector["Min_end"] = df_sector["Min"] + 0.01
                        df_sector["Max_end"] = df_sector["Max"] + 0.01

                        bars = (
                            alt.Chart(df_sector)
                            .mark_bar(color="#4BA3FF")
                            .encode(
                                x=alt.X("Sector:N", title="", sort="-y"),
                                y=alt.Y("Weight:Q", title="Weight in Equity"),
                                tooltip=[
                                    alt.Tooltip("Sector:N"),
                                    alt.Tooltip("Weight:Q", format=".2%"),
                                ],
                            )
                        )

                        min_marks = (
                            alt.Chart(df_sector)
                            .mark_bar(color="yellow", opacity=0.9)
                            .encode(
                                x=alt.X("Sector:N", title="", sort="-y"),
                                y=alt.Y("Min:Q"),
                                y2=alt.Y2("Min_end:Q"),
                                tooltip=[alt.Tooltip("Min:Q", title="Min", format=".2%")],
                            )
                            .transform_filter("datum.Min != null")
                        )

                        max_marks = (
                            alt.Chart(df_sector)
                            .mark_bar(color="red", opacity=0.9)
                            .encode(
                                x=alt.X("Sector:N", title="", sort="-y"),
                                y=alt.Y("Max:Q"),
                                y2=alt.Y2("Max_end:Q"),
                                tooltip=[alt.Tooltip("Max:Q", title="Max", format=".2%")],
                            )
                            .transform_filter("datum.Max != null")
                        )

                        chart_sector = (bars + min_marks + max_marks).properties(
                            height=300
                        ).interactive()

                        st.altair_chart(chart_sector, use_container_width=True)
                    else:
                        st.info("No equity allocation in the current solution.")

                # --- ESG (within Equity) ---
                with colC:
                    st.markdown("**ESG Breakdown (within Equity)**")

                    if not esg_in_eq.empty:
                        df_esg = esg_in_eq.reset_index()
                        df_esg.columns = ["ESG", "Weight"]

                        df_esg["Min"] = df_esg["ESG"].map(
                            lambda s: esg_constraints.get(s, {}).get("min", None)
                            if esg_constraints
                            else None
                        )
                        df_esg["Max"] = df_esg["ESG"].map(
                            lambda s: esg_constraints.get(s, {}).get("max", None)
                            if esg_constraints
                            else None
                        )

                        df_esg["Min_end"] = df_esg["Min"] + 0.01
                        df_esg["Max_end"] = df_esg["Max"] + 0.01

                        bars_esg = (
                            alt.Chart(df_esg)
                            .mark_bar(color="#4BA3FF")
                            .encode(
                                x=alt.X("ESG:N", title=""),
                                y=alt.Y("Weight:Q", title="Weight in Equity"),
                                tooltip=[
                                    alt.Tooltip("ESG:N"),
                                    alt.Tooltip("Weight:Q", format=".2%"),
                                ],
                            )
                        )

                        min_marks_esg = (
                            alt.Chart(df_esg)
                            .mark_bar(color="yellow", opacity=0.9)
                            .encode(
                                x=alt.X("ESG:N", title=""),
                                y=alt.Y("Min:Q"),
                                y2=alt.Y2("Min_end:Q"),
                                tooltip=[alt.Tooltip("Min:Q", title="Min", format=".2%")],
                            )
                            .transform_filter("datum.Min != null")
                        )

                        max_marks_esg = (
                            alt.Chart(df_esg)
                            .mark_bar(color="red", opacity=0.9)
                            .encode(
                                x=alt.X("ESG:N", title=""),
                                y=alt.Y("Max:Q"),
                                y2=alt.Y2("Max_end:Q"),
                                tooltip=[alt.Tooltip("Max:Q", title="Max", format=".2%")],
                            )
                            .transform_filter("datum.Max != null")
                        )

                        chart_esg = (bars_esg + min_marks_esg + max_marks_esg).properties(
                            height=300
                        ).interactive()

                        st.altair_chart(chart_esg, use_container_width=True)
                    else:
                        st.info("No equity allocation in the current solution.")

                st.markdown(
                    "<span style='color:#DAA520;'>Yellow horizontal markers</span> "
                    "indicate **minimum allocation bounds**, while "
                    "<span style='color:red;'>red horizontal markers</span> indicate "
                    "**maximum allocation limits** for each sector, ESG bucket or asset class.",
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # ---------- Within each non-equity asset class ----------
                if within_non_eq:
                    st.markdown("### Composition Within Non-Equity Asset Classes")

                    for ac_name, series_ac in within_non_eq.items():
                        st.markdown(f"**{ac_name} – allocation within this asset class**")
                        df_ac_within = series_ac.reset_index()
                        df_ac_within.columns = ["Name", "WithinClassWeight"]

                        chart_ac_within = (
                            alt.Chart(df_ac_within)
                            .mark_bar()
                            .encode(
                                x=alt.X("Name:N", sort="-y", title="Instrument", axis=alt.Axis(labelAngle=0)),
                                y=alt.Y(
                                    "WithinClassWeight:Q",
                                    title="Weight within this asset class",
                                ),
                                tooltip=[
                                    alt.Tooltip("Name:N", title="Instrument"),
                                    alt.Tooltip(
                                        "WithinClassWeight:Q",
                                        title="Weight in class",
                                        format=".2%",
                                    ),
                                ],
                            )
                        ).properties(height=250)

                        st.altair_chart(chart_ac_within, use_container_width=True)

                # ---------- Full table ----------
                with st.expander("Full Portfolio Weights"):
                    df_full = today_df.copy()
                    df_full["Weight"] = df_full["Weight"].map(lambda x: f"{x:.2%}")
                    st.dataframe(
                        df_full[["TICKER", "NAME", "ASSET_CLASS", "SECTOR", "ESG", "Weight"]],
                        use_container_width=True,
                    )


# --------------- PAGE 3: AI Assistant ---------------
def page_ai_assistant():

    # HERO + GLOBAL STYLE
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

.hero-ai {
  background: linear-gradient(120deg, #133c55 0%, #1d4e6e 55%, #2a628f 100%);
  height: 200px;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 32px 46px;
  margin-bottom: 40px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
  font-family: 'Open Sans', sans-serif;
}

.hero-ai-title {
  color: #ffffff !important;
  font-size: 42px;
  font-weight: 700;
  margin: 0;
}

.hero-ai-text {
  margin-top: 10px;
  color: #d9dde7;
  font-size: 18px;
  max-width: 1200px;
  line-height: 1.6;
}

/* Pillars */
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  margin-top: 24px;
  margin-bottom: 24px;
}

.pill-card {
  flex: 1 1 260px;
  background: #f3f4f6;
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 4px 12px rgba(15,23,42,0.06);
}

.pill-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 8px;
  color: #111827;
}

.pill-body {
  font-size: 16px;
  color: #4b5563;
  line-height: 1.7;
}

/* PARAGRAPH STYLE */
.paragraph {
  font-size: 18px;
  color: #111827;
  line-height: 1.9;
  margin-bottom: 24px;
  max-width: 1200px;
}

.paragraph strong {
  font-weight: 600;
}

</style>



<div class="hero-ai">
  <div class="hero-ai-title">Phi Assistant</div>
  <div class="hero-ai-text">
    Your personal quantitative guide – helping you understand the platform, 
    financial concepts, and market context with clarity and professionalism.
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

    # EXPLANATION TEXT

    st.markdown(
        """
<div class="paragraph">
Our AI-powered assistant is designed to enhance your experience on our portfolio management platform. 
It can support you across three key areas:
</div>
        """,
        unsafe_allow_html=True,
    )

    # THREE PILLARS
    st.markdown(
        """
<div class="pill-row">
  <div class="pill-card">
    <div class="pill-title">1. Platform Guidance & Functionality</div>
    <div class="pill-body">
      <strong>Navigate the application with confidence:</strong><br><br>
      The assistant can explain each section of the site:
      from selecting your investment universe to configuring 
      constraints, interpreting backtests, and reviewing today’s 
      optimal portfolio. Whether you're unsure about a step or want to
      understand how a specific feature works, it provides clear, client-friendly guidance.
    </div>
  </div>
  <div class="pill-card">
    <div class="pill-title">2. Financial & Theoretical Concepts</div>
    <div class="pill-body">
      <strong>Understand the rationale behind your portfolio:</strong><br><br>
      Ask about diversification, risk/return trade-offs, risk aversion, the Markowitz optimization 
      framework, or the meaning of any chart or metric shown in the application. The assistant provides clear 
      explanations grounded in quantitative finance - always educational, never advisory.
    </div>
  </div>
  <div class="pill-card">
    <div class="pill-title">3. Market Context & Current Themes</div>
    <div class="pill-body">
      <strong>Stay informed about what’s happening in the financial world:</strong><br><br>
      You can request neutral, factual insights about current market conditions, macroeconomic themes, 
      or asset-class developments. The assistant offers high-level context to help you make sense of the 
      broader environment in which portfolios operate.
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

    # DISCLAIMER
    st.markdown(
        """
<div class="paragraph">
⚠️ <strong>Important:<strong> The assistant provides general information and educational insights only. 
It does <strong>not<strong> offer personalized investment recommendations or specific trading advice.
</div>
        """,
        unsafe_allow_html=True,
    )

    client = get_llm_client()

    # Initialize chat history in session_state
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "system",
                "content": (
                    """
                    You are Phi Assistant — the digital investment assistant integrated into Phi Investment Capital’s portfolio management platform.
                    
                    Your purpose is to help users understand how the platform works, explain investment and quantitative concepts, and provide neutral market context. You are always educational and explanatory, never advisory.
                    
                    =====================
                    1. Your identity
                    =====================
                    - You represent **Phi Investment Capital**, a quantitative portfolio design firm.
                    - Your tone is **professional, calm, and client-oriented**, similar to a relationship manager or investment consultant at an asset management firm.
                    - You assist with **understanding and navigation**, not with making investment decisions for the user.
                    
                    =====================
                    2. What the platform contains
                    =====================
                    
                    The application has several main pages:
                    
                    • **About Us**
                      - Describes Phi Investment Capital, its quantitative philosophy, who it serves, and its focus on transparency, discipline, and explainable portfolio construction.
                    
                    • **Our Investment Approach**
                      - Explains that the platform uses long-only, fully invested **Markowitz mean–variance optimisation**.
                      - Mentions estimation windows, shrinkage covariance, rebalancing frequency, risk-aversion (gamma), ESG and sector integration, asset-class constraints, transaction costs, and tiered management fees.
                      - Includes a clear assumptions and disclaimers section: backtests are hypothetical, not guarantees of future performance.
                    
                    • **Portfolio Optimization**
                      This page guides the user through a 5-step workflow to build and analyse a portfolio:
                      1) **General settings** – choose equity universe (S&P 500 or MSCI World), investment amount, investment horizon, estimation window, and rebalancing frequency.
                      2) **Universe & filters** – apply sector and ESG filters to equities; select other asset classes (e.g. bonds, commodities, alternatives) and specific instruments.
                      3) **Risk profile questionnaire** – a set of questions that produces a risk score and an internal risk-aversion parameter (gamma).
                      4) **Constraints** – set maximum weight per asset, sector constraints, ESG constraints, asset-class constraints at total portfolio level, and optional turnover limits.
                      5) **Optimisation & backtest** – run a constrained, long-only mean–variance optimisation, simulate a backtest with rebalancing, transaction costs and tiered management fees, and display performance charts (cumulative return, wealth evolution), drawdowns, and summary statistics (volatility, Sharpe, max drawdown, fees, turnover).
                    
                    • **Current Optimal Allocation**
                      - Shows the latest optimised portfolio given the user’s configuration.
                      - Displays top holdings, allocation by asset class, equity sector breakdown, and equity ESG breakdown.
                      - Helps the user understand how the portfolio is currently positioned under their chosen assumptions and constraints.
                    
                    • **Phi Assistant**
                      - The page where you (Phi Assistant) live.
                      - You help with three pillars:
                        1) Platform guidance & functionality (how to use each page and step).
                        2) Financial & theoretical concepts (diversification, risk/return, Markowitz, gamma, drawdowns, etc.).
                        3) Market context & current themes (neutral, factual, high-level discussion of markets and asset classes).
                    
                    • **Become a Client**
                      - Explains the onboarding process and shows a simple multi-step description (submit details, describe objectives, receive confirmation).
                      - Contains an application form where users enter information about themselves/their organisation, objectives, risk attitude, preferred portfolio style, investment size, and any key constraints.
                      - After submission, the user receives a confirmation email summarising their information.
                    
                    • **Meet the Team**
                      - Presents the core team members with their roles (e.g. Quant & Portfolio Strategy, CIO & Quantitative Research, CFO, client-facing roles, etc.).
                      - Highlights the blend of quantitative expertise, portfolio engineering, and client focus behind the platform.
                      - Team members: Illia Shapkin (Quant & Portfolio Strategy) , Rean Morinaj  (CIO & Quantitative Research), Malek Trimeche (Client Relations), Petrit Gashi (CFO - Finance & Operations)

                      
                    
                    =====================
                    3. What you should help with
                    =====================
                    
                    You primarily help users understand:
                    
                    • **Platform functionality**
                      - Explain what each page and step does.
                      - Clarify what different inputs mean (e.g. risk questionnaire, ESG filters, sector constraints, asset-class choices, turnover limits, estimation window, rebalancing frequency).
                      - Describe, in general terms, how the optimiser and backtest work, without assuming you see the user’s exact numerical inputs unless they tell you.
                    
                    • **Optimizer outputs and analytics**
                      - Explain conceptually what charts and metrics represent:
                        - Cumulative return and wealth evolution.
                        - Annualised volatility and Sharpe ratio.
                        - Maximum drawdown and recovery.
                        - Turnover, transaction costs, and management fees.
                        - Allocation by asset class, sector, and ESG bucket.
                      - Help the user interpret these outputs in an educational way (e.g. “this metric shows…”, “this chart illustrates…”), not as investment advice.
                    
                    • **Quantitative and financial concepts**
                      - Diversification, correlation, and concentration risk.
                      - Risk/return trade-offs and the role of risk aversion (gamma).
                      - How constraints (sector, ESG, asset-class, max weights) shape the feasible set of portfolios.
                      - The meaning of long-only, fully invested portfolios and rebalancing.
                      - The assumptions and limitations of backtests and models.
                    
                    • **Market context (high-level only)**
                      - Neutral, factual explanations of asset classes (equities, bonds, commodities, etc.).
                      - High-level descriptions of market regimes or macro themes when asked.
                      - No predictions, no timing calls, no security-specific recommendations.
                    
                    =====================
                    4. Behaviour and style
                    =====================
                    
                    When responding:
                    
                    • Be clear, structured, and concise.  
                    • Use plain language with optional light technical detail when helpful.  
                    • When a question relates to a specific app feature, mention which page or step it belongs to and how the user can interact with it.  
                    • Emphasise transparency and discipline: explain that users control assumptions (universe, constraints, fees, horizon, etc.) and the engine applies those rules consistently.  
                    • If the user seems confused, break the answer into simple parts and, if relevant, suggest where in the app they can see or change that setting.
                    
                    =====================
                    5. Compliance and limitations
                    =====================
                    
                    You **must not**:
                    - Provide personalised investment advice or recommendations.
                    - Tell users what they should buy, sell, or hold.
                    - Suggest specific positions, target returns, or forecasts for individual securities or portfolios.
                    - Claim that backtested performance guarantees or reliably indicates future results.
                    
                    You **may**:
                    - Explain trade-offs in general terms (e.g. “higher expected return typically comes with higher volatility”).
                    - Describe the implications of different choices inside the app (e.g. tighter constraints, different universes, shorter vs longer estimation windows).
                    - Encourage users to think in terms of risk tolerance, diversification, and long-term discipline.
                    
                    Always maintain a neutral, educational stance. Your goal is to help users understand the platform and the concepts behind it, not to direct their investments.
                    
                    =====================
                    5. Additional info:
                     - Company contacts:
                    Email: phinvestments@hotmail.com
                    Address: 123 Financial Street, Geneva, Switzerland
                    Phone: +41 22 123 45 67
                     - to become a client the person have to check page "Become a Client" and fill the request or contact directly
                     - the platform doesn't ahve any account creation possiblity so don't talk about it at all
                     
                    =====================
                    
                    """
                ),
            }
        ]

    # Show previous messages (except system)
    for msg in st.session_state.ai_messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question")
    if user_input:
        # 1) Add user message to history
        st.session_state.ai_messages.append({"role": "user", "content": user_input})

        # 2) Display user bubble
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3) Call OpenAI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=st.session_state.ai_messages,
                )

                reply = response.choices[0].message.content
                st.markdown(reply)

        # 4) Save assistant reply in history
        st.session_state.ai_messages.append(
            {"role": "assistant", "content": reply}
        )


def page_investment_approach():
    import streamlit as st

    # Global styles for this page
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Open Sans', sans-serif;
}

/* HERO */
.hero-approach {
  background: linear-gradient(120deg, #133c55 0%, #1d4e6e 55%, #2a628f 100%);
  min-height: 200px;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 32px 46px;
  margin-bottom: 40px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
}

.hero-approach-title {
  color: #ffffff;
  font-size: 40px;
  font-weight: 700;
  margin: 0;
}

.hero-approach-text {
  margin-top: 10px;
  color: #d9dde7;
  font-size: 18px;
  max-width: 1200px;
  line-height: 1.6;
}

/* TYPOGRAPHY */
.section-heading {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 10px;
  color: #101827;
  margin-top: 30px;
}

.paragraph {
  font-size: 18px;
  color: #111827;
  line-height: 1.9;
  margin-bottom: 24px;
  max-width: 900px;
}

.paragraph strong {
  font-weight: 600;
}

/* STEP GRID */
.step-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
  margin: 12px 0 28px 0;
}

.step-card {
  background: #f9fafb;
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}

.step-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 999px;
  background: #133c55;
  color: #ffffff;
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 8px;
}

.step-title {
  font-size: 17px;
  font-weight: 600;
  margin-bottom: 6px;
  color: #111827;
}

.step-body {
  font-size: 15px;
  color: #4b5563;
  line-height: 1.7;
}

/* FEE TABLE */
.fee-section {
  margin-top: 10px;
  margin-bottom: 30px;
}

.fee-table {
  width: 100%;
  max-width: 640px;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 16px;
}

.fee-table thead {
  background: #e5e7eb;
}

.fee-table th,
.fee-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #e5e7eb;
  text-align: left;
}

.fee-table th {
  font-weight: 600;
  color: #111827;
}

.fee-table td {
  color: #374151;
}

.fee-pill-note {
  font-size: 15px;
  color: #4b5563;
  margin-top: 8px;
}

/* INFO BOXES */
.info-box {
  margin-top: 24px;
  padding: 18px 20px;
  border-radius: 14px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  max-width: 900px;
}

.info-title {
  font-size: 17px;
  font-weight: 600;
  margin-bottom: 6px;
  color: #1e3a8a;
}

.info-body {
  font-size: 15px;
  color: #1f2937;
  line-height: 1.8;
}

/* DISCLAIMER BOX */
.disclaimer-box {
  margin-top: 30px;
  padding: 18px 20px;
  border-radius: 14px;
  background: #fef3c7;
  border: 1px solid #facc15;
  max-width: 900px;
}

.disclaimer-title {
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 6px;
  color: #92400e;
}

.disclaimer-body {
  font-size: 15px;
  color: #78350f;
  line-height: 1.8;
}

/* SMALL TAGLINE */
.approach-tagline {
  margin-top: 40px;
  font-style: italic;
  font-size: 15px;
  color: #6b7280;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # Hero
    st.markdown(
        """
<div class="hero-approach">
  <div class="hero-approach-title">Our Investment Approach</div>
  <div class="hero-approach-text">
    We build long-only, multi-asset portfolios using a transparent, rules-based process that combines
    quantitative research with practical implementation. Every assumption is visible. Every decision
    is explainable.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Intro
    st.markdown(
        """
<div class="paragraph">
At Phi Investment Capital, we believe that long-term results are driven less by short-term forecasts and more by
<strong>structure, diversification, and discipline</strong>. Our platform converts institutional-grade portfolio
engineering into a clear, intuitive workflow that investors can understand, explain, and reuse.
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 1 – Universe & preferences (with step cards)
    st.markdown('<div class="section-heading">1. From Universe to Client Preferences</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="paragraph">
Every portfolio begins with a well-defined investment universe and a precise understanding of the client’s
preferences. Our process is designed to make these choices explicit rather than implicit.
</div>

<div class="step-grid">
  <div class="step-card">
    <div class="step-badge">01</div>
    <div class="step-title">Select the investment universe</div>
    <div class="step-body">
      Clients choose between major equity universes such as the S&amp;P 500 or MSCI World, and may extend the
      portfolio with complementary asset classes such as bonds, commodities, or alternative exposures.
    </div>
  </div>

  <div class="step-card">
    <div class="step-badge">02</div>
    <div class="step-title">Define the risk profile</div>
    <div class="step-body">
      A concise risk questionnaire translates preferences into a quantitative risk-aversion parameter.
      This determines how the optimiser balances expected return versus volatility and drawdowns.
    </div>
  </div>

  <div class="step-card">
    <div class="step-badge">03</div>
    <div class="step-title">Set ESG and sector preferences</div>
    <div class="step-body">
      Clients can include or exclude ESG buckets, constrain sector exposures, and apply limits at the equity
      level. These preferences become hard constraints within the optimisation.
    </div>
  </div>

  <div class="step-card">
    <div class="step-badge">04</div>
    <div class="step-title">Configure asset-class constraints</div>
    <div class="step-body">
      For multi-asset portfolios, minimum and maximum weights can be set by asset class (e.g. fixed income,
      commodities, alternatives), aligning the portfolio with investment guidelines or top-down views.
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 2 – Portfolio construction
    st.markdown('<div class="section-heading">2. Robust Portfolio Construction</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="paragraph">
Once preferences are defined, we construct portfolios using a long-only, fully invested version of
<strong>Markowitz mean–variance optimisation</strong>, enhanced with techniques commonly used in institutional practice.
</div>

<div class="step-grid">
  <div class="step-card">
    <div class="step-badge">05</div>
    <div class="step-title">Risk estimation</div>
    <div class="step-body">
      We use rolling estimation windows and shrinkage covariance estimators to obtain stable, noise-reduced
      measures of risk and correlation between assets.
    </div>
  </div>

  <div class="step-card">
    <div class="step-badge">06</div>
    <div class="step-title">Constraints & implementability</div>
    <div class="step-body">
      The optimiser respects per-asset caps, ESG and sector limits, asset-class constraints, and optional
      turnover bounds. Portfolios are constructed to be diversified, long-only, and fully invested.
    </div>
  </div>

  <div class="step-card">
    <div class="step-badge">07</div>
    <div class="step-title">Disciplined rebalancing</div>
    <div class="step-body">
      Portfolios are rebalanced on a monthly, quarterly, or yearly schedule. This maintains diversification,
      keeps constraints satisfied, and avoids reactive, emotion-driven trading.
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 3 – Fees & costs
    st.markdown('<div class="section-heading">3. Management Fees and Trading Costs</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="paragraph">
We believe that costs should be modelled as carefully as returns. Our backtests incorporate both transaction
costs and a transparent, tiered management fee schedule.
</div>

<div class="fee-section">
  <div class="paragraph"><strong>Tiered annual management fee (based on initial wealth):</strong></div>
  <table class="fee-table">
    <thead>
      <tr>
        <th>Initial Wealth (CHF)</th>
        <th>Annual Fee</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Below 10 million</td>
        <td><strong>0.50%</strong> p.a.</td>
      </tr>
      <tr>
        <td>10m – 20m</td>
        <td><strong>0.45%</strong> p.a.</td>
      </tr>
      <tr>
        <td>20m – 30m</td>
        <td><strong>0.40%</strong> p.a.</td>
      </tr>
      <tr>
        <td>30m – 50m</td>
        <td><strong>0.35%</strong> p.a.</td>
      </tr>
      <tr>
        <td>50m – 100m</td>
        <td><strong>0.30%</strong> p.a.</td>
      </tr>
      <tr>
        <td>Above 100 million</td>
        <td><strong>0.25%</strong> p.a.</td>
      </tr>
    </tbody>
  </table>
  <div class="fee-pill-note">
    In the backtest, the applicable annual fee is converted into a monthly rate and deducted progressively
    over time. This allows clients to distinguish clearly between gross and net performance.
  </div>
</div>

<div class="info-box">
  <div class="info-title">Transaction costs and turnover</div>
  <div class="info-body">
    Rebalancing incurs proportional transaction costs that depend on portfolio turnover. Optional turnover
    constraints can be applied to control trading intensity, balancing implementation costs against
    responsiveness to new information.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 4 – Backtesting & current allocation
    st.markdown('<div class="section-heading">4. Historical Backtests and Today’s Allocation</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="paragraph">
Our platform provides both a historical lens and a current snapshot, using the exact same portfolio
construction logic in each case.
</div>

<div class="info-box">
  <div class="info-title">Historical backtests</div>
  <div class="info-body">
    Backtests show how a given configuration would have behaved in past market environments, including cumulative
    returns, volatility, drawdowns, turnover, fees paid, and comparisons against benchmark indices. This helps
    investors understand the behaviour of a portfolio before committing capital.
  </div>
</div>

<div class="info-box">
  <div class="info-title">Current optimal allocation</div>
  <div class="info-body">
    Using the latest available data, the platform computes today’s optimal portfolio under the same rules
    and constraints. Clients can see current weights by asset, sector, asset class, and ESG bucket, alongside
    the key risk and diversification characteristics of the portfolio.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 5 – Assumptions & disclaimers
    st.markdown('<div class="section-heading">5. Assumptions and Important Disclaimers</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="paragraph">
Transparency extends to the limitations of any model. We state clearly what our simulations capture
and what they omit.
</div>

<div class="disclaimer-box">
  <div class="disclaimer-title">Assumptions</div>
  <div class="disclaimer-body">
    • Portfolios are long-only and fully invested with no leverage.<br>
    • Historical market data is used for estimation; future returns are unknown and not forecast.<br>
    • Transaction costs and management fees are modelled in a simplified but consistent way.<br>
    • Taxes, market impact, and liquidity frictions beyond standard assumptions are not included.<br>
    • Benchmarks are used for comparison only and are not directly investable.
  </div>
</div>

<div class="disclaimer-box" style="margin-top:16px;">
  <div class="disclaimer-title">Important notice</div>
  <div class="disclaimer-body">
    Backtested results are hypothetical and do not represent actual trading or realised performance.
    They are not a guarantee of future results and should not be relied upon as personalised investment
    advice or a solicitation to buy or sell any security. The platform is designed for illustration,
    education, and portfolio design under user-specified assumptions.
  </div>
</div>

<div class="approach-tagline">
  Our commitment is to make every assumption explicit, every decision reproducible, and every portfolio
  understandable.
</div>
        """,
        unsafe_allow_html=True,
    )




def page_new_client():

    BREVO_API_KEY = st.secrets["brevo"]["api_key"]
    SENDER_EMAIL = st.secrets["brevo"]["sender_email"]
    SENDER_NAME = st.secrets["brevo"]["sender_name"]

    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
      font-family: 'Open Sans', sans-serif;
    }

    /* HERO */
    .hero-client {
      background: linear-gradient(120deg, #133c55 0%, #1d4e6e 55%, #2a628f 100%);
      min-height: 190px;
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 28px 40px;
      margin-bottom: 32px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }

    .hero-client-title {
      color: #ffffff;
      font-size: 34px;
      font-weight: 700;
      margin: 0;
    }

    .hero-client-subtitle {
      margin-top: 10px;
      color: #d9dde7;
      font-size: 17px;
      max-width: 1200px;
      line-height: 1.6;
    }

    /* PARAGRAPH */
    .paragraph {
      font-size: 18px;
      color: #111827;
      line-height: 1.9;
      margin-bottom: 18px;
      max-width: 1200px;
    }

    .paragraph strong {
      font-weight: 600;
    }

    /* HORIZONTAL TIMELINE (3 STEPS) */
    .client-steps-timeline {
      position: relative;
      display: flex;
      gap: 32px;
      margin-top: 18px;
      margin-bottom: 26px;
      padding-top: 24px;
    }

    .client-steps-timeline::before {
      content: "";
      position: absolute;
      top: 32px;
      left: 0;
      right: 0;
      height: 2px;
      background: #e5e7eb;
      z-index: 0;
    }

    .client-step-item {
      flex: 1;
      text-align: center;
    }

    .client-step-circle {
      width: 32px;
      height: 32px;
      border-radius: 999px;
      background: #133c55;
      color: #ffffff;
      font-size: 14px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 8px auto;
      position: relative;
      z-index: 1;
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.25);
    }

    .client-step-title {
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 4px;
      color: #111827;
    }

    .client-step-body {
      font-size: 14px;
      color: #4b5563;
      line-height: 1.7;
      max-width: 320px;
      margin: 0 auto;
    }

    /* SMALL LABEL ABOVE FORM */
    .form-label {
      font-size: 25px;
      font-weight: 600;
      margin-top: 10px;
      margin-bottom: 6px;
      color: #374151;
    }
    </style>

    <div class="hero-client">
      <div class="hero-client-title">Become a Client</div>
      <div class="hero-client-subtitle">
        Start your onboarding with Phi Investment Capital. Share your profile, objectives, and preferences so we can
        assess whether our portfolio solutions are a good fit for your needs.
      </div>
    </div>

    <div class="paragraph">
    Welcome to our <strong>Client Application Portal</strong>. The information you provide below helps us understand
    your investment profile, risk tolerance, and constraints. We use these details to review your request and, if
    appropriate, to design a portfolio proposal aligned with your objectives.
    </div>

    <div class="client-steps-timeline">

      <div class="client-step-item">
        <div class="client-step-circle">1</div>
        <div class="client-step-title">Submit your details</div>
        <div class="client-step-body">
          Provide basic information about you or your organisation, your investment horizon, and your approximate
          allocation size.
        </div>
      </div>

      <div class="client-step-item">
        <div class="client-step-circle">2</div>
        <div class="client-step-title">Describe your objectives</div>
        <div class="client-step-body">
          Indicate your risk attitude, preferred portfolio style (equity-focused or multi-asset), and any key
          constraints or preferences we should respect.
        </div>
      </div>

      <div class="client-step-item">
        <div class="client-step-circle">3</div>
        <div class="client-step-title">Receive a confirmation</div>
        <div class="client-step-body">
          After submission, you will receive a confirmation email summarising your information. Our team can then
          review your profile and contact you with next steps if appropriate.
        </div>
      </div>

    </div>

    <div class="form-label">
      Application form: 
    </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 1) GENERAL INFORMATION
    # ------------------------------------------------------------------
    st.subheader("General Information")

    full_name = st.text_input("Full Name", key="client_full_name")
    email = st.text_input("Email Address", key="client_email")
    phone = st.text_input("Phone Number (optional)", key="client_phone")
    over_18 = st.checkbox("I confirm that I am at least 18 years old", key="client_over_18")

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2) INVESTMENT PREFERENCES
    # ------------------------------------------------------------------
    st.subheader("Investment Preferences")

    backtest_config = st.session_state.get("backtest_results", None)
    has_opt_config = isinstance(backtest_config, dict)

    opt_summary_text = ""
    reuse_option_available = False

    if has_opt_config:
        try:
            r = backtest_config

            universe_choice_label = "S&P 500" if r.get("universe_choice") == "SP500" else "MSCI World"
            rebal_map = {12: "Yearly", 3: "Quarterly", 1: "Monthly"}
            rebal_label = rebal_map.get(r.get("rebalancing"), "Custom")
            selected_ac = r.get("selected_asset_classes_other") or []
            ac_str = ", ".join(selected_ac) if selected_ac else "Equity only"

            est_months = r.get("est_months", 12)
            max_w = r.get("max_weight_per_asset", 0.05)
            max_to = r.get("max_turnover_per_rebalance", None)

            constraints_summary = build_constraints_summary(r)
            opt_summary_text = f"""
- **Planned investment amount:** {r.get("investment_amount"):,.0f}  
- **Equity universe:** {universe_choice_label}  
- **Investment horizon:** {r.get("investment_horizon_years")} years  
- **Rebalancing frequency:** {rebal_label}  
- **Estimation window:** {est_months} months  
- **Risk profile:** {r.get("profile_label")}  
- **Max weight per asset:** {max_w:.0%}  
- **Turnover constraint:** {"No explicit turnover limit" if max_to is None else f"Max {max_to:.0%} one-way turnover per rebalance"}  
- **Additional asset classes:** {ac_str}  

{constraints_summary}
"""
            reuse_option_available = True
        except Exception:
            reuse_option_available = False

    if reuse_option_available:
        pref_source = st.radio(
            "How would you like to define your investment preferences?",
            options=[
                "Use my latest Portfolio Optimization settings",
                "Specify my preferences here",
            ],
            index=0,
            help=(
                "If you have already configured and run a portfolio optimization, you can reuse "
                "those settings as your investment preferences. Otherwise, specify them below."
            ),
        )
    else:
        st.info(
            "No previous portfolio optimization settings were found in this session, "
            "or they are not available. Please specify your investment preferences below."
        )
        pref_source = "Specify my preferences here"

    manual_prefs = {}

    if pref_source == "Specify my preferences here":
        st.markdown("Please give us a high-level view of how you would like to invest.")

        col1, col2 = st.columns(2)
        with col1:
            manual_prefs["universe_choice"] = st.radio(
                "Preferred equity universe",
                options=["S&P 500", "MSCI World"],
                index=0,
            )
            manual_prefs["horizon"] = st.selectbox(
                "Investment horizon",
                options=["1–3 years", "3–5 years", "5–7 years", "7+ years"],
                index=1,
            )
        with col2:
            manual_prefs["risk_profile"] = st.selectbox(
                "Risk profile",
                options=[
                    "Very Conservative",
                    "Conservative",
                    "Balanced",
                    "Dynamic",
                    "Aggressive",
                ],
                index=2,
            )
            manual_prefs["multi_asset"] = st.selectbox(
                "Preferred portfolio type",
                options=[
                    "Equity-focused",
                    "Multi-asset (Equity, Fixed Income, Commodities, Alternatives)",
                ],
                index=1,
            )

        manual_prefs["investment_amount"] = st.number_input(
            "Planned investment amount",
            min_value=1_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
            help="Approximate amount you are considering investing."
        )

        manual_prefs["notes"] = st.text_area(
            "Additional comments or preferences (optional)",
            placeholder=(
                "Example: I prefer a diversified portfolio with limited drawdowns, "
                "I am interested in sustainable investing, I'd like to have at least 50% in equity, etc."
            ),
        )

    else:
        st.markdown(
            "You chose to **reuse your latest Portfolio Optimization settings**. "
            "Below is a summary of the preferences that will be used in your application:"
        )
        st.markdown(opt_summary_text)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 3) SUBMIT APPLICATION & SEND EMAIL
    # ------------------------------------------------------------------
    submit = st.button("Submit Application", type="primary")

    if submit:
        if not full_name or not email:
            st.warning("Please enter at least your full name and email address.")
            return
        if not over_18:
            st.warning("You must confirm that you are at least 18 years old to proceed.")
            return

        greeting = f"Dear {full_name},\n\n"
        intro = (
            "Thank you for your request to become a client of Phi Investment Capital.\n"
            "We are pleased to receive your application and appreciate your interest in our "
            "quantitative asset and risk management approach.\n\n"
        )

        general_info_block = "Your general information:\n"
        general_info_block += f"- Full Name: {full_name}\n"
        general_info_block += f"- Email: {email}\n"
        if phone:
            general_info_block += f"- Phone: {phone}\n"

        # Preferences block
        if pref_source == "Specify my preferences here":
            prefs_block = "Your investment preferences (as specified in the form):\n"
            prefs_block += f"- Planned investment amount: {manual_prefs['investment_amount']:,.0f}\n"
            prefs_block += f"- Preferred equity universe: {manual_prefs['universe_choice']}\n"
            prefs_block += f"- Investment horizon: {manual_prefs['horizon']}\n"
            prefs_block += f"- Risk profile: {manual_prefs['risk_profile']}\n"
            prefs_block += f"- Portfolio type: {manual_prefs['multi_asset']}\n"
            if manual_prefs["notes"]:
                prefs_block += f"- Additional comments: {manual_prefs['notes']}\n"
            prefs_block += "\n"
            constraints_block = ""
        else:
            r = backtest_config
            universe_choice_label = "S&P 500" if r.get("universe_choice") == "SP500" else "MSCI World"
            rebal_map = {12: "Yearly", 3: "Quarterly", 1: "Monthly"}
            rebal_label = rebal_map.get(r.get("rebalancing"), "Custom")
            selected_ac = r.get("selected_asset_classes_other") or []
            ac_str = ", ".join(selected_ac) if selected_ac else "Equity only"
            est_months = r.get("est_months", 12)
            max_w = r.get("max_weight_per_asset", 0.05)
            max_to = r.get("max_turnover_per_rebalance", None)

            prefs_block = "\nYour investment preferences (based on your latest portfolio optimization settings):\n\n"
            prefs_block += f"- Planned investment amount: {r.get('investment_amount'):,.0f}\n"
            prefs_block += f"- Equity universe: {universe_choice_label}\n"
            prefs_block += f"- Investment horizon: {r.get('investment_horizon_years')} years\n"
            prefs_block += f"- Rebalancing frequency: {rebal_label}\n"
            prefs_block += f"- Estimation window: {est_months} months\n"
            prefs_block += f"- Risk profile: {r.get('profile_label')}\n"
            prefs_block += f"- Max weight per asset: {max_w:.0%}\n"
            prefs_block += f"- Turnover constraint: {'No explicit turnover limit' if max_to is None else f'Max {max_to:.0%} one-way turnover per rebalance'}\n"
            prefs_block += f"- Additional asset classes: {ac_str}\n\n"

            constraints_block = format_constraints_block(r)

        closing = (
            "We will review your information and contact you in the near future to discuss the next steps.\n"
            "This email is an acknowledgement of your request and does not constitute investment advice "
            "or an offer to purchase or sell any financial instrument.\n\n"
            "Best regards,\n"
            "Phi Investment Capital\n"
        )

        full_email_text = greeting + intro + general_info_block + prefs_block
        if constraints_block:
            full_email_text += constraints_block + "\n\n"
        full_email_text += closing

        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = BREVO_API_KEY

        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
            sib_api_v3_sdk.ApiClient(configuration)
        )

        email_content = sib_api_v3_sdk.SendSmtpEmail(
            sender={"name": SENDER_NAME, "email": SENDER_EMAIL},
            to=[{"email": email}],
            subject="Phi Investment Capital – Client Application Confirmation",
            text_content=full_email_text,
        )

        try:
            # IMPORTANT: keep using the full email object, not just the address
            api_instance.send_transac_email(email_content)
            st.success(f"Your application has been submitted and a confirmation email was sent to {email}.")
        except ApiException as e:
            # Combine both str(e) and e.body (if present) to search the text
            body = getattr(e, "body", "")
            combined = (str(e) + " " + str(body)).lower()

            if "email is not valid" in combined or "invalid_parameter" in combined:
                st.error("Please enter a valid email address.")
            else:
                st.error("Something went wrong while sending your message. Please try again.")

def page_our_team():
    petrit_img = img_to_base64("Petrit.jpg")
    rean_img = img_to_base64("Rean.jpg")
    illia_img = img_to_base64("illia.jpg")
    malek_img = img_to_base64("Malek.jpg")

    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
      font-family: 'Open Sans', sans-serif;
    }

    /* HERO */
    .hero-team {
      background: linear-gradient(120deg, #133c55 0%, #1d4e6e 55%, #2a628f 100%);
      min-height: 180px;
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 28px 40px;
      margin-bottom: 32px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }

    .hero-team-title {
      color: #ffffff;
      font-size: 34px;
      font-weight: 700;
      margin: 0;
    }

    .hero-team-subtitle {
      margin-top: 10px;
      color: #d9dde7;
      font-size: 17px;
      max-width: 900px;
      line-height: 1.6;
    }

    /* PARAGRAPH */
    .paragraph {
      font-size: 18px;
      color: #111827;
      line-height: 1.9;
      margin-bottom: 18px;
      max-width: 1100px;
    }

    .paragraph strong {
      font-weight: 600;
    }

    /* TEAM GRID */
    .team-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 26px;
      margin-top: 28px;
    }

    /* MEMBER CARD */
    .team-member {
      background: #f9fafb;
      border-radius: 18px;
      padding: 18px 18px 20px 18px;
      text-align: center;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .team-member:hover {
      transform: translateY(-3px);
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.14);
    }

    .team-member img {
      width: 160px;
      height: 160px;
      object-fit: cover;
      border-radius: 999px;
      margin-bottom: 12px;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }

    .member-name {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 4px;
      color: #111827;
    }

    .member-role {
      font-size: 14px;
      color: #6b7280;
      margin-bottom: 8px;
    }

    .member-bio {
      font-size: 14px;
      color: #374151;
      margin-bottom: 10px;
      line-height: 1.6;
    }

    /* LinkedIn button */
    .linkedin a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid #0a66c2;
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      color: #0a66c2;
    }

    .linkedin a:hover {
      background: #0a66c2;
      color: #ffffff;
    }
    </style>

    <div class="hero-team">
      <div class="hero-team-title">Meet the Team</div>
      <div class="hero-team-subtitle">
        A multidisciplinary team combining quantitative research, portfolio engineering, and client expertise.
        We bring together rigorous analysis and real-world investment experience to build tools that serve
        allocators with clarity and precision.
      </div>
    </div>

    <div class="paragraph">
    Behind Phi Investment Capital is a compact, hands-on team. Each member contributes a distinct mix of
    quantitative skills, market experience, and product thinking - all focused on turning complex portfolio
    construction into something transparent, disciplined, and client-ready.
    </div>
        """,
        unsafe_allow_html=True,
    )

    html = f"""
    <div class="team-grid">
        <div class="team-member">
            <img src="data:image/jpeg;base64,{illia_img}">
            <div class="member-name">Illia Shapkin</div>
            <div class="member-role">Quant & Portfolio Strategy</div>
            <div class="member-bio">Focused on risk analysis and optimization. Data-driven decision maker.</div>
            <div class="linkedin"><a href="https://www.linkedin.com/in/illia-shapkin-b1956b226/" target="_blank">LinkedIn</a></div>
        </div>
        <div class="team-member">
            <img src="data:image/jpeg;base64,{rean_img}">
            <div class="member-name">Rean Morinaj</div>
            <div class="member-role">CIO & Quantitative Research</div>
            <div class="member-bio">Specializes in modeling, stats and robust optimization logic. 
            Responsible for an organization's technology strategy and systems</div>
            <div class="linkedin"><a href="https://www.linkedin.com/in/rean-morinaj/" target="_blank">LinkedIn</a></div>
        </div>
        <div class="team-member">
            <img src="data:image/jpeg;base64,{malek_img}">
            <div class="member-name">Malek Trimeche</div>
            <div class="member-role">Client Relations</div>
            <div class="member-bio">Analytical and client-focused. Building bridges between finance and people.</div>
            <div class="linkedin"><a href="https://www.linkedin.com/in/malek-trimeche-4616b8352/" target="_blank">LinkedIn</a></div>
        </div>
        <div class="team-member">
            <img src="data:image/jpeg;base64,{petrit_img}">
            <div class="member-name">Petrit Gashi</div>
            <div class="member-role">CFO - Finance & Operations</div>
            <div class="member-bio">Manages financial oversight, operational controls, and resource planning to ensure long-term 
            stability and efficient execution across the firm.</div>
            <div class="linkedin"><a href="https://www.linkedin.com/in/petrit-gashi-8433a9289/" target="_blank">LinkedIn</a></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
