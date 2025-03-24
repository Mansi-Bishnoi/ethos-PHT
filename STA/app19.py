# --- Quartile Visualization ---
st.subheader("📊 Quartile Distribution of Event Times")

# Compute deciles (Q1 to Q10)
quantiles = np.quantile(timeline_df["Time"], np.linspace(0.1, 1.0, 10))

# Create a DataFrame for visualization
quantile_df = pd.DataFrame({"Quantile": [f"Q{i}" for i in range(1, 11)], "Time": quantiles})

# Plot using Box Plot
fig_box = px.box(timeline_df, y="Time", points="all", title="Event Time Distribution with Quartiles")

# Add quantile markers
for i, q in enumerate(quantiles):
    fig_box.add_hline(y=q, line_dash="dash", line_color="red", annotation_text=f"Q{i+1}: {q:.2f}")

st.plotly_chart(fig_box)
