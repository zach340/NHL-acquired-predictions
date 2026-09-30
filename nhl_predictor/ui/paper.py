"""Research-paper pages shown from the sidebar (static content)."""

import os

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def introduction():
    st.subheader("Introduction")
    st.caption("An abstract and overview of the project.")
    st.markdown(
        """
        For this project I wanted to build something that could actually help answer a question that I think a lot of hockey fans and even front offices are not fully sure how to approach, which is how accurately can historical NHL data predict offensive performance for forwards and defensive impact for defensemen and how can those predictions be used to project how a player might perform if they were to change teams.
        
        The reason I wanted to focus on this specifically is because I think there is a lot of uncertainty around how a player is going to perform when they change teams and while the numbers can tell us a lot they do not always give us the full picture, so I wanted to see if building a model around historical data could help close that gap.
        
        The tool is designed to be used by NHL GMs, fantasy hockey players, and anyone who is into hockey analytics because I wanted it to be something that felt useful across different types of people who care about the game in different ways. It uses a machine learning model trained on historical NHL data to predict performance for any player so that when a trade or signing happens the data is already there to project how they are going to do in a new context. It also includes a contract evaluator that takes those projections and uses age curves to assess whether a player's performance is actually worth what they are being paid, which I thought was important to add because predicting performance only tells part of the story if you are a GM or a fantasy player trying to make a decision. On top of that there is a player search interface and a validation section that compares the model's predictions against live NHL API data from the current season so you can see how well it is actually working.
        ---

        **Project:** NHL Player Predictor
        How accurately can historical NHL data predict offensive performance for forwards and defensive impact for defensemen, and how can those predictions be used to project how a player might perform if they were to change teams?


        """
    )


def literature_review():
    st.subheader("Literature Review")
    st.caption("An overview of existing research and sources relevant to this project.")
    st.markdown(
        """
        There is actually a decent amount of work out there that touches on different pieces of what this project is trying to do, even if nothing does all of it together in one place.
        
        The most directly relevant starting point is the NHL EDGE stats breakdown of Mikko Rantanen's outlook after his trade to the Hurricanes published on NHL.com. What I liked about this piece is that it goes pretty deep on how Rantanen performed with Colorado and what his underlying numbers looked like before the trade. The abeck2309 NHL trade ROI project connects to this well because it takes that same idea and builds a model around it, using realized and expected goals above replacement to put an actual number on whether a trade worked out. The problem with both of them though is they are looking backwards at what already happened and neither one is projecting how a player is going to do going forward in a new system with new linemates, which is exactly the gap this project is trying to close.
        
        On the data side, the Hockey Analytics article on pulling data directly from the NHL API was really useful because it walks through how to actually get the data without having to worry about a lot of the finer details of scraping. This connects to something like the AAZZAZRON TradeTracker project which is a Discord bot that scrapes Sportsnet to pull in recent trade and signing details automatically. That one is a little out of scope for where the project is right now but is worth keeping in mind for future versions when financial data becomes more important.
        
        The most important thing to note is that all of this data needs to be validated and cleaned before it can be used in the model. This is where the NHL API comes in, as it provides a reliable source of real-time data that can be used to verify the accuracy of the historical data.
        
        Beyond those specific sources there is a broader body of hockey analytics research that is relevant here. Work on expected goals models and wins above replacement in hockey has shown that raw counting stats like goals and assists do not always tell the full story of how valuable a player actually is, which is part of why this project focuses on underlying production metrics rather than just point totals. There is also research on how team context affects individual performance, specifically how linemates, zone deployment, and coaching systems can inflate or suppress a player's numbers in ways that make it hard to project them into a new situation. This is the core challenge the model is trying to account for. Finally there is existing work on age curves in hockey which shows that forwards typically peak in their mid-twenties and decline gradually after that, which is the foundation for how the contract evaluator in this project works.
        """
    )


def methodology():
    st.subheader("Research Methodology")
    st.caption("A walkthrough of the specific techniques and methods used in this project.")
    
    st.markdown("""
    *This section describes the specific techniques and methods used, connecting methodology
    directly to the Research Question. It includes methods of data collection, analysis, and
    the choices made to refine or limit the project.*
    """)

    st.markdown("---")
    st.markdown("#### Data Collection")
    st.markdown("""
    The data for this project comes from moneypuck.com which has fully downloadable historical NHL data going back to the 2008-2009 season all the way up to 2024-2025, and I validated it using the NHL API to make sure what I was working with was accurate. Moneypuck also provides lines data which I used to better understand the context of each player's performance, and it does have up to date stats available as well, but I chose not to include those yet because I wanted to see how the model performed on historical data first and then use the API to validate those predictions before adding anything else.

    I split forwards and defensemen into separate datasets because they have very different roles and I wanted to look for different primary stats for each, so keeping them together would have made the model less accurate and would have added a lot of stats that were not relevant to what I was actually trying to predict. From there I pulled out the offensive stats that I felt had the most impact on points and goals per game and fed those into the model, cutting out anything that was not essential so the datasets stayed manageable. I also filtered out players who did not reach a minimum number of games or minutes played because I wanted to make sure I was only working with players who actually had a real sample size and not guys who only played a handful of games.
    """)

    st.markdown("#### Data Management")
    st.markdown("""
    One thing I had to figure out early on was what to do with missing values in the data. Rather than just dropping those rows entirely I converted any NAs to zero because since all the features are numerical a missing stat is basically the same as zero production in that category, and deleting the whole row would have thrown away a lot of valid data that was still useful. I also joined age data to the main dataset using player ID and season as the merge keys rather than player name because names can have spelling variations and special characters especially for international players, so using the ID just made it cleaner and avoided mismatches. To keep the app running smoothly I also saved the trained models as joblib files so they load directly at runtime instead of retraining every single time someone opens the page.
    """)

    st.markdown("---")

    # ── Technical breakdown PDF download ─────────────────────────────────────
    _pdf_path = os.path.join(ROOT, "analysis_technical.pdf")
    if os.path.exists(_pdf_path):
        with open(_pdf_path, "rb") as _pdf_file:
            st.download_button(
                label="📄 Technical Breakdown — download full model documentation (PDF)",
                data=_pdf_file,
                file_name="analysis_technical.pdf",
                mime="application/pdf",
                help="In-depth technical documentation: LightGBM details, feature weights, "
                     "residual modeling, cross-validation setup, and training parameters.",
            )

    st.markdown("#### Analysis & Modeling")
    st.markdown("""
                This section explains how the prediction model works and you do not need a technical background to follow along, and if you do want the more technical side of things you can download the PDF above.
                """)
    st.markdown("**What does it do?**")
    st.markdown("""
                You put in a player's history and skill profile and the model estimates how many points, goals, and overall contributions they are likely to put up, both in general and specifically on any of the 32 NHL teams. I built separate models for forwards and defensemen because they have very different roles and I wanted to make sure the model was actually looking for the right things for each.
                """)
    st.markdown("**What is it actually predicting?**")
    st.markdown("""
                Rather than just predicting a raw stat line the model is trying to figure out something more useful, which is whether a player is going to outperform or underperform their own historical baseline. Every player gets a personal benchmark built from their career history and recent seasons and the model's job is to figure out whether their skills and team fit are going to push them above or below that mark. I thought this was a better approach than just predicting that good players score more which is not really telling you anything useful.
                """)
    st.markdown("**What information goes in?**")
    st.markdown("""
                The model pulls from four main areas. First is shooting and skill, so how dangerous is this player's shot, how often are they beating goalies relative to what you would expect, and how much are they contributing on the power play. Second is career history, so what have they done across their career, what did last season look like, and whether they are trending up or down. Third is age and career stage because a 25 year old and a 33 year old with the same stats are in very different situations and the model accounts for that. Fourth is team and system fit because ice time, line quality, and shot generation all vary a lot by organization and the model can swap in any of the 32 teams to simulate how a player would do in a different system, which is really the core of what makes this useful for trades and signings.
                """)
    st.markdown("**Two model versions**")
    st.markdown("""
                The Team Fit model is for right now. You give it a player's current skill profile and it tells you how they would produce on each team today, so it is best for trade deadline and free agency decisions. The Next Season model is for planning ahead because it adds trajectory signals like year over year stat changes to figure out whether a player is still improving or starting to decline, which makes it better for long term contracts and draft planning.
                """)

    st.markdown("**How does it work?**")
    st.markdown("""
                The model uses machine learning algorithms to analyze the input data and identify patterns and relationships. It then applies these patterns to make predictions about future performance.
                """)
    st.markdown("**How is it trained and validated?**")
    st.markdown("""
                The model gets tested by training on some historical seasons and predicting others, rotating three times so every data point gets a chance to be evaluated. I also made a few deliberate choices to make sure it is most accurate where it actually matters, so players with very few games are left out to avoid noisy samples and elite players are weighted more heavily during training so the model gets sharper at the predictions that actually affect roster decisions the most.
                """)
    st.markdown("---")
    st.markdown("#### Choices & Limitations")
    st.markdown("""Goalies and low-minutes players were excluded from the model because they don't
    provide a reliable sample for predictions — goalies in particular use entirely
    different performance metrics. A broader limitation is the inherent unpredictability
    of the NHL itself. Players and teams are constantly evolving, coaches and systems
    change, and those shifts can affect performance in ways the model can't fully
    anticipate. The model's accuracy is strong, but it's not perfect, and there will
    always be factors outside the data.

    Contract dollar values were also left out. Partly this came down to time — finding
    a clean, free salary datasource proved difficult — but more fundamentally, each
    team values players differently based on their own needs and circumstances. Without
    a reliable way to model that context, adding salary data risked making the
    contract recommendations less accurate rather than more.
    """)

    st.markdown("---")
    st.markdown("#### AI Tool Usage")
    st.markdown("""
    Claude was used throughout development to help with code generation, debugging,
    and model design. Its ability to produce working code quickly was especially valuable
    given the volume of Streamlit logic involved, and it was a useful guide for someone
    with limited prior Streamlit experience.

    That said, there were real limitations. Claude didn't always understand exactly
    what data was available or how it needed to be structured, which meant generated
    code often required manual verification and adjustment. There were also points where
    Claude's suggested approach to the model differed from what the data actually
    supported — in early testing the model had a performance cap that was suppressing
    true high-end predictions, and catching that required understanding the underlying
    data well enough to recognize the problem. AI assistance works best when you still
    know what the right answer should look like.
    """)


def analysis_findings():
    st.subheader("Analysis & Findings")
    st.caption("What was discovered through the analysis.")

    st.markdown("#### Key Findings")
    st.markdown("""
    Overall I think the model performed pretty well. Across 427 matched players the Points/GP MAE came in at 0.123 and the Goals/GP MAE came in at 0.074 which means on average the model is off by about an eighth of a point per game on points and less than a tenth of a goal per game on goals. For a model trying to predict individual player production across an entire season I think that is a solid result.

    The prediction spread ratio came in at 95% which means the model is generating predictions that cover a realistic range rather than just clustering everything in the middle, and the calibration slope of 0.90 means it is tracking pretty closely with how players actually performed with just a slight tendency to be conservative on the higher end.
    """)

    st.markdown("#### Elite Player Predictions")
    st.markdown("""
    The trickiest part of the model was predicting elite players and that shows up in the numbers. For the top 9% of players the Points/GP MAE goes up to 0.154 with a bias of -0.102, and the Goals/GP MAE goes up to 0.105 with a bias of -0.075. The negative bias means the model is consistently undershooting on elite players which makes sense because truly elite production is harder to predict from historical data alone. This is something I would want to improve in future versions, but given that elite players were already weighted three times more heavily during training I think this is about as good as you can get without adding more contextual data.
    """)

    st.markdown("#### Defensive Metrics")
    st.markdown("""
    Looking at the scatter plots the model does best on Hits/Game with an MAE of 0.258 and a correlation of 0.89 which is really strong and shows the model is picking up on physical play patterns well. Takeaways/Game and PIM/Game were harder to predict with correlations of 0.60 and 0.58 respectively, which honestly makes sense because those stats are a lot more random and situational than something like hits. A player can take a penalty or win a puck battle in ways that are hard to predict from historical trends alone so I was not too surprised to see the model struggle a bit more there.
    """)

    st.markdown("#### Surprises & Anomalies")
    st.markdown("""
    The biggest surprise was how well the model handled the spread of predictions. I was worried early on that it was going to cluster everything around the mean and not really differentiate between players, which is a common problem with this type of model. The 95% prediction spread ratio shows that it is not doing that which I think is the most important thing to get right if you actually want to use this for roster decisions. The elite underprediction is the main thing I would flag as something to keep an eye on because if you are a GM trying to evaluate a superstar trade that bias could matter.
    """)


def conclusion():
    st.subheader("Conclusion")
    st.caption("A summary of what was built and what it means.")

    st.markdown("""
    Going back to the original research question of how accurately can historical NHL data predict player performance and how can those predictions be used to project how a player might perform if they change teams, I think the honest answer is that this project does not answer that question directly but instead builds something that lets anyone answer it themselves.
    """)

    st.markdown("""
    What I built is a tool that takes a player's historical data and gives you a projection of how they would perform on any of the 32 NHL teams right now or going into next season. On top of that the contract evaluator takes those performance projections and uses the player's age curve to project how long a contract should be, which I think is one of the most practically useful parts of the whole app because knowing how a player will perform is only half the decision. The other half is figuring out how long you can count on that level of production, and that is what the contract side is trying to help with.
    """)

    st.markdown("""
    So rather than me running a single study on one trade and calling the question answered, anyone using this app can plug in any player and get that answer for whatever situation they are actually looking at. I think that is more useful in practice because the question is going to look different depending on whether you are a GM at the trade deadline, a fantasy player making a pickup, or just someone curious about what would happen if a player switched teams.
    """)

    st.markdown("""
    The model's validation results show that it is accurate enough to be genuinely useful and the elite player underprediction is something I would want to keep working on. But the foundation is there and the goal from the start was to build something that could be a real tool for real decisions, and I think that is what it is.
    """)


def works_cited():
    st.subheader("Works Cited")
    st.caption("Sources used throughout this project.")

    # MLA 9th edition — one flat alphabetical list, hanging-indent style via HTML.
    _cite = (
        "<p style='margin:0 0 1.1em 0; padding-left:2em; text-indent:-2em; "
        "font-size:15px; line-height:1.6;'>{}</p>"
    )

    st.markdown(
        # 1. AAZZAZRON (AA)
        _cite.format(
            "AAZZAZRON. \"TradeTracker: A Discord Bot That Scrapes Sportsnet to Find "
            "the Most Recent NHL Trades and Signings.\" <em>GitHub</em>, "
            "<a href='https://github.com/AAZZAZRON/TradeTracker'>"
            "github.com/AAZZAZRON/TradeTracker</a>. Accessed 2025."
        ) +
        # 2. abeck2309 (AB)
        _cite.format(
            "abeck2309. \"nhl-trade-roi-xgar: Evaluating NHL Trades Using Realized "
            "and Expected xGAR.\" <em>GitHub</em>, "
            "<a href='https://github.com/abeck2309/nhl-trade-roi-xgar'>"
            "github.com/abeck2309/nhl-trade-roi-xgar</a>. Accessed 2025."
        ) +
        # 3. Anthropic (AN)
        _cite.format(
            "Anthropic. <em>Claude</em>, version claude-sonnet-4-20250514, "
            "Anthropic, 2025, "
            "<a href='https://claude.ai'>claude.ai</a>. Accessed 2025."
        ) +
        # 4. "Hockey Analytics..." (H)
        _cite.format(
            "\"Hockey Analytics &ndash; Getting Data Directly from the NHL API.\" "
            "<em>Hockey-Statistics</em>, 14 May 2025, "
            "<a href='https://hockey-statistics.com/2025/05/14/hockey-analytics-getting-data-directly-from-the-nhl-api/'>"
            "hockey-statistics.com/2025/05/14/hockey-analytics-getting-data-directly-from-the-nhl-api/</a>."
        ) +
        # 5. MoneyPuck (M)
        _cite.format(
            "MoneyPuck. \"MoneyPuck.com.\" <em>MoneyPuck</em>, "
            "<a href='https://moneypuck.com'>moneypuck.com</a>. Accessed 2025."
        ) +
        # 6. National Hockey League (NA)
        _cite.format(
            "National Hockey League. \"NHL Stats API.\" <em>NHL</em>, "
            "<a href='https://api-web.nhle.com'>api-web.nhle.com</a>. Accessed 2025."
        ) +
        # 7. "NHL EDGE Stats..." (NH)
        _cite.format(
            "\"NHL EDGE Stats: Rantanen's Outlook after Trade to Hurricanes.\" "
            "<em>NHL</em>, "
            "<a href='https://www.nhl.com/news/edge-stats-impact-of-trade-on-mikko-rantanen-martin-necas'>"
            "www.nhl.com/news/edge-stats-impact-of-trade-on-mikko-rantanen-martin-necas</a>. Accessed 2025."
        ),
        unsafe_allow_html=True,
    )


PAGES = {
    "Introduction": introduction,
    "Literature Review": literature_review,
    "Methodology": methodology,
    "Analysis & Findings": analysis_findings,
    "Conclusion": conclusion,
    "Works Cited": works_cited,
}
