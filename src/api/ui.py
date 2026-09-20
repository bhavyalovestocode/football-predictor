from html import escape


def dashboard_html(feature_columns, team_names=()):
    feature_fields = "\n".join(
        f'''<label class="field"><span>{escape(column)}</span>'''
        f'''<input name="{escape(column)}" type="number" step="any" value="0" required></label>'''
        for column in feature_columns
    )
    team_options = "\n".join(
        f'<option value="{escape(team, quote=True)}">{escape(team)}</option>'
        for team in team_names
    )

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UCL Match Probability Desk</title>
  <style>
    :root { --ink: #172121; --muted: #66706e; --paper: #f5f2eb; --panel: #fffdf8; --line: #d9d4c8; --accent: #df5a3d; --green: #2d7d65; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: radial-gradient(circle at 90% 0%, #f0c7a8 0, transparent 27rem), var(--paper); font: 15px/1.45 Georgia, serif; }
    main { max-width: 1180px; margin: 0 auto; padding: 42px 22px 64px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 30px; }
    h1 { margin: 0; max-width: 560px; font-size: clamp(2rem, 5vw, 4.5rem); line-height: .95; letter-spacing: -.04em; font-weight: 500; }
    .kicker { color: var(--accent); font: 700 12px/1.2 Arial, sans-serif; letter-spacing: .12em; text-transform: uppercase; }
    .lede { max-width: 290px; color: var(--muted); margin: 0; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 20px; align-items: start; }
    section { background: var(--panel); border: 1px solid var(--line); padding: 22px; box-shadow: 8px 8px 0 rgba(23, 33, 33, .06); }
    h2 { font-size: 1.2rem; margin: 0 0 16px; }
    .fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 11px 14px; max-height: 640px; overflow: auto; padding-right: 6px; }
    .team-panel { border-bottom: 1px solid var(--line); padding-bottom: 20px; margin-bottom: 20px; }
    .team-fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    select { width: 100%; border: 1px solid var(--line); background: #fff; color: var(--ink); padding: 10px; font: 14px Georgia, serif; }
    select:focus { outline: 2px solid #efb29d; border-color: var(--accent); }
    .field { display: grid; gap: 4px; color: var(--muted); font: 11px Arial, sans-serif; }
    input { width: 100%; border: 1px solid var(--line); background: #fff; color: var(--ink); padding: 9px 10px; font: 14px Georgia, serif; }
    input:focus { outline: 2px solid #efb29d; border-color: var(--accent); }
    .controls { display: flex; gap: 12px; align-items: end; margin-top: 20px; }
    .threshold { max-width: 150px; }
    button { border: 0; background: var(--ink); color: white; padding: 12px 18px; cursor: pointer; font: 700 13px Arial, sans-serif; }
    button:hover { background: var(--accent); }
    .results { position: sticky; top: 20px; }
    .status { color: var(--muted); min-height: 24px; margin: 0 0 15px; font: 13px Arial, sans-serif; }
    .cards { display: grid; gap: 10px; }
    .card { border-left: 5px solid var(--line); padding: 13px 15px; background: #f8f5ef; }
    .card.home { border-color: #df5a3d; } .card.draw { border-color: #d2a33b; } .card.away { border-color: var(--green); }
    .card-head { display: flex; justify-content: space-between; font: 700 12px Arial, sans-serif; text-transform: uppercase; }
    .prob { font-size: 2rem; margin: 5px 0 0; } .odds { color: var(--muted); font: 12px Arial, sans-serif; }
    .outcome { margin-top: 18px; padding: 14px; background: var(--ink); color: white; text-align: center; }
    .outcome strong { display: block; color: #f5c6a9; font-size: 1.4rem; margin-top: 4px; }
    @media (max-width: 800px) { header { display: block; } .lede { margin-top: 15px; } .layout { grid-template-columns: 1fr; } .results { position: static; } }
    @media (max-width: 480px) { main { padding: 26px 14px 40px; } section { padding: 16px; } .fields { grid-template-columns: 1fr; } .controls { align-items: stretch; flex-direction: column; } .threshold { max-width: none; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><div class="kicker">UEFA Champions League / calibrated model</div><h1>Match probability desk</h1></div>
      <p class="lede">Enter the pre-match history features to see the model's probability split and decimal odds.</p>
    </header>
    <div class="layout">
      <section>
        <div class="team-panel">
          <h2>Choose a matchup</h2>
          <form id="team-form">
            <div class="team-fields">
              <label class="field"><span>Home team</span><select id="home-team" required><option value="">Select home team</option>__TEAM_OPTIONS__</select></label>
              <label class="field"><span>Away team</span><select id="away-team" required><option value="">Select away team</option>__TEAM_OPTIONS__</select></label>
            </div>
            <div class="controls"><label class="field threshold"><span>Draw threshold</span><input id="team-draw-threshold" type="number" min="0" max="1" step="0.01" value="0.26"></label><button type="submit">Predict matchup</button></div>
          </form>
        </div>
        <h2>Manual feature desk</h2>
        <form id="prediction-form">
          <div class="fields">__FEATURE_FIELDS__</div>
          <div class="controls">
            <label class="field threshold"><span>Draw threshold</span><input id="draw-threshold" type="number" min="0" max="1" step="0.01" value="0.26"></label>
            <button type="submit">Calculate prediction</button>
          </div>
        </form>
      </section>
      <section class="results">
        <h2>Prediction</h2>
        <p id="status" class="status">Ready for a feature row.</p>
        <div class="cards">
          <div class="card home"><div class="card-head"><span>Home win</span><span id="home-odds">-</span></div><div id="home-prob" class="prob">-</div><div class="odds">estimated probability</div></div>
          <div class="card draw"><div class="card-head"><span>Draw</span><span id="draw-odds">-</span></div><div id="draw-prob" class="prob">-</div><div class="odds">estimated probability</div></div>
          <div class="card away"><div class="card-head"><span>Away win</span><span id="away-odds">-</span></div><div id="away-prob" class="prob">-</div><div class="odds">estimated probability</div></div>
        </div>
        <div class="outcome">Predicted outcome<strong id="outcome">-</strong></div>
      </section>
    </div>
  </main>
  <script>
    const form = document.getElementById('prediction-form');
    const teamForm = document.getElementById('team-form');
    const status = document.getElementById('status');
    const formatProbability = value => `${(value * 100).toFixed(1)}%`;
    const formatOdds = value => `Odds ${value.toFixed(2)}`;
    const renderResult = result => {
      document.getElementById('home-prob').textContent = formatProbability(result.probabilities.home_win);
      document.getElementById('draw-prob').textContent = formatProbability(result.probabilities.draw);
      document.getElementById('away-prob').textContent = formatProbability(result.probabilities.away_win);
      document.getElementById('home-odds').textContent = formatOdds(result.implied_odds.home_win);
      document.getElementById('draw-odds').textContent = formatOdds(result.implied_odds.draw);
      document.getElementById('away-odds').textContent = formatOdds(result.implied_odds.away_win);
      document.getElementById('outcome').textContent = result.predicted_outcome;
    };
    teamForm.addEventListener('submit', async event => {
      event.preventDefault();
      status.textContent = 'Loading latest team states...';
      try {
        const response = await fetch('/predict/teams', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({home_team: document.getElementById('home-team').value, away_team: document.getElementById('away-team').value, draw_threshold: Number(document.getElementById('team-draw-threshold').value)}) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Prediction failed.');
        renderResult(result);
        status.textContent = 'Team matchup prediction updated.';
      } catch (error) { status.textContent = error.message; }
    });
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const features = Object.fromEntries(new FormData(form).entries());
      delete features['draw-threshold'];
      Object.keys(features).forEach(key => { features[key] = Number(features[key]); });
      status.textContent = 'Calculating...';
      try {
        const response = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({features, draw_threshold: Number(document.getElementById('draw-threshold').value)}) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Prediction failed.');
        renderResult(result);
        status.textContent = 'Prediction updated.';
      } catch (error) { status.textContent = error.message; }
    });
  </script>
</body>
</html>"""
    return template.replace("__FEATURE_FIELDS__", feature_fields).replace(
        "__TEAM_OPTIONS__", team_options
    )