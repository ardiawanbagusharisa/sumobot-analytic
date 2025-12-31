from streamlit_modal import Modal
import pandas as pd
import streamlit as st
from plotting.stoc import stoc
import os

from plotting.overall_analyzer import (
    show_overall_analysis,
    plot_full_cross_heatmap_half
)

from plotting.individual_analyzer import (
    show_individual_report
)

from PIL import Image

@st.cache_data
def load_summary_data():
    summarized_dir = "result"

    df_sum = pd.read_csv(f"{summarized_dir}/summary_bot.csv").rename(columns={"Duration": "Duration (ms)"})
    df = pd.read_csv(f"{summarized_dir}/summary_matchup.csv")
    df_timebins = pd.read_csv(f"{summarized_dir}/summary_action_timebins.csv")
    df_collision_timebins = pd.read_csv(f"{summarized_dir}/summary_collision_timebins.csv")
    return df_sum, df, df_timebins, df_collision_timebins

def load_arena_data(dfsum):
    # Check if arena_heatmap directory exists
    arena_heatmaps_output = "result/arena_heatmaps"

    if os.path.exists(arena_heatmaps_output):
        # Get all bot directories
        bot_dirs = [d for d in os.listdir(arena_heatmaps_output)
                   if os.path.isdir(os.path.join(arena_heatmaps_output, d))]

        # Sort bot directories by rank from dfsum
        if "Rank" in dfsum.columns and "Bot" in dfsum.columns:
            rank_map = dfsum.groupby("Bot")["Rank"].first().to_dict()
            bot_dirs = sorted(bot_dirs, key=lambda b: rank_map.get(b, 9999))
        else:
            # Fallback to alphabetical sorting if rank columns not found
            bot_dirs = sorted(bot_dirs)

        if bot_dirs:
            phase_names = ["window_2.5-15s.png", "window_15-30s.png", "window_30-45s.png","window_45-60s.png"]

            # Display heatmaps for each bot
            for bot_name in bot_dirs:
                toc.h3(f"{bot_name} (#{bot_dirs.index(bot_name)+1})")
                bot_dir = os.path.join(arena_heatmaps_output, bot_name)

                # Create n columns for the n phases
                cols = st.columns(len(phase_names))

                for col, phase_name in zip(cols, phase_names):
                    image_path = os.path.join(bot_dir, f"{phase_name}")

                    with col:
                        st.markdown(f"**{phase_name}**")
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            st.image(image, use_container_width=True)
                        else:
                            st.warning(f"Image not found: {phase_name}")

                # Display position distribution
                dist_path = os.path.join(bot_dir, "position_distribution.png")
                if os.path.exists(dist_path):
                    st.markdown("**Position Distribution (X & Y Overlayed)**")
                    dist_image = Image.open(dist_path)
                    st.image(dist_image, use_container_width=True)

                # Display distance distribution
                dist_path = os.path.join(bot_dir, "distance_distribution.png")
                if os.path.exists(dist_path):
                    st.markdown("**Distance Distribution**")
                    dist_image = Image.open(dist_path)
                    st.image(dist_image, use_container_width=True)

                st.markdown("**Full Configuration Analysis**")
                fig = plot_full_cross_heatmap_half(df, bot_name=bot_name)
                st.pyplot(fig)
                
                st.divider()
        else:
            st.warning("No bot heatmaps found in directory")
            st.info("Run: `python detailed_analyzer.py all` to generate heatmaps")
    else:
        st.warning(f"Heatmap directory not found: {arena_heatmaps_output}")
        st.info("Run: `python detailed_analyzer.py all` to generate heatmaps for all bots")

if __name__ == "__main__":
    toc = stoc()

    df_sum, df, df_timebins, df_collision_timebins  = load_summary_data()
    df_sum = df_sum.rename(columns={"Duration":"Duration (ms)"})

    st.set_page_config(page_title="Sumobot Performance Dashboard")
    width = st.sidebar.slider("plot width", 1, 25, 8)
    height = st.sidebar.slider("plot height", 1, 25, 6)

    cfg = {
        "Timer": sorted(df["Timer"].unique().tolist()),
        "ActInterval": sorted(df["ActInterval"].unique().tolist()),
        "Round": sorted(df["Round"].unique().tolist()),
        "SkillLeft": sorted(df["SkillLeft"].unique().tolist()),
        "SkillRight": sorted(df["SkillRight"].unique().tolist()),
        "Bots": sorted(df["Bot_L"].unique().tolist()),
    }
    bots = str.join(", ", cfg["Bots"])

    toc.h1("Sumobot Performance Dashboard")
    st.markdown("A quick visual overview of bot performance metrics across matchups, timers, and actions")
    st.markdown(f"This experiment conducted with bots: {bots}")
    st.markdown("Configuration :" )
    st.write(cfg)
    st.markdown("Source code: https://github.com/arbyazra123/sumobot_data")

    # Summary
    toc.h2("Summary Matchup")
    st.dataframe(df_sum)

    modal = Modal("Complete Matchup", key="matchup")
    if st.button("View complete matchup"):
        modal.open()

    if modal.is_open():
        with modal.container():
            st.dataframe(df, use_container_width=True, hide_index=True)

    show_overall_analysis(df,cfg,df_timebins, df_collision_timebins,toc,width,height)

    show_individual_report(df,toc,width,height)

    # Arena Heatmaps
    toc.h2("Arena Heatmaps - Bot Movement Analysis")
    st.markdown("Visualize bot movement patterns across different game phases (Early, Mid, Late)")
    load_arena_data(df_sum)

    toc.toc()

