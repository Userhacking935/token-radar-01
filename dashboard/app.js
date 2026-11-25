// Token Radar's dashboard. It reads window.RADAR (written beside dashboard/ by main.ts) and
// renders the ranked board, filters, a per-protocol detail view with the model's own arithmetic,
// the farm calculator and the live-token set. No network: every byte it shows is in data.js.
//
// How a row's numbers are made (the same arithmetic main.ts runs, so the page and board.json
// agree): the value regression predicts a token's market cap from ln(TVL), seven-day growth,
// protocol age, chain count and category; conditional FDV divides it by the initial float
// (20% of supply, a stated assumption); launch odds come from a logistic fit on all 758
// protocols; implied value is odds x FDV. A farm plan prices a position at its best size.
(function () {
  'use strict';

  var DATA = window.RADAR;
  if (!DATA) return;
  var app = document.getElementById('board');
  var state = { q: '', category: 'all', verdict: 'all', watch: false, tab: 'pre-token', sort: 'implied', dir: -1, selected: null };

  function usd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }
  function usdFull(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return '$' + Math.round(n).toLocaleString('en-US');
  }
  function pct(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n > 0 ? '+' : '') + n.toFixed(2) + '%';
  }
  function ageFmt(days) {
    if (days === null || days === undefined) return 'unknown';
    if (days >= 365) return (days / 365).toFixed(1) + 'y';
    return Math.max(0, Math.round(days)) + 'd';
  }
  function oddsFmt(p) { return (p * 100).toFixed(1) + '%'; }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k.slice(0, 2) === 'on') node.addEventListener(k.slice(2), attrs[k]);
        else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i];
      if (child === null || child === undefined) continue;
      if (Array.isArray(child)) child.forEach(function (c) { if (c) node.appendChild(c); });
      else if (typeof child === 'string') node.appendChild(document.createTextNode(child));
      else node.appendChild(child);
    }
    return node;
  }

  // The odds model's features, rebuilt here so the page can draw the hazard curve and the
  // farm calculator with the fitted coefficients, not with numbers typed in by hand.
  var ODDS_CATS = ['Dexs', 'Liquid Staking', 'Lending', 'CEX', 'Bridge', 'Risk Curators', 'Yield', 'RWA'];
  function oddsFeatures(p, ageDays) {
    var ageY = ageDays / 365;
    var g = p.change_7d === null ? 0 : Math.min(120, Math.max(-60, p.change_7d));
    var x = [1, Math.log(p.tvl_usd), ageY, ageY * ageY, g / 100, Math.log(Math.max(1, p.chains.length))];
    ODDS_CATS.forEach(function (c) { x.push(p.category === c ? 1 : 0); });
    return x;
  }
  function oddsAt(p, ageDays) {
    var b = DATA.odds_fit.coefficients;
    var z = oddsFeatures(p, ageDays).reduce(function (s, v, i) { return s + v * b[i]; }, 0);
    return 1 / (1 + Math.exp(-z));
  }
  function thetaFor(category) {
    var c = (category || '').toLowerCase();
    if (c.indexOf('dex') >= 0 || c.indexOf('perp') >= 0 || c.indexOf('deriv') >= 0) return 0.075;
    if (c.indexOf('rwa') >= 0) return 0.035;
    return 0.05;
  }
  function holdFor(age) {
    if (age === null || age === undefined) return 45;
    if (age < 180) return 60;
    if (age < 365) return 45;
    return 30;
  }
  function farmAt(p, capital) {
    var hold = holdFor(p.age_days);
    var weeks = hold / 7;
    var pool = p.launch_odds * p.conditional_fdv_usd * thetaFor(p.category);
    var share = Math.min(capital / p.tvl_usd, 0.005);
    var expected = pool * share;
    // Gas only: the snapshot carries no yield data, so no opportunity rate is invented.
    var cost = 8 + 2 * weeks + 3;
    var hours = 0.25 + 0.17 * weeks + 0.17;
    var net = expected - cost;
    var perHour = net / hours;
    // Verdict by launch-odds tier (a stated assumption), only with a positive expected net.
    var verdict = p.launch_odds >= 0.6 && net > 0 ? 'FARM' : p.launch_odds >= 0.35 && net > 0 ? 'MARGINAL' : 'SKIP';
    return { expected: expected, cost: cost, hours: hours, net: net, perHour: perHour, verdict: verdict };
  }

  function readHash() {
    var hash = window.location.hash.replace(/^#/, '');
    if (!hash) return;
    var params = new URLSearchParams(hash);
    if (params.get('q')) state.q = params.get('q');
    if (params.get('category')) state.category = params.get('category');
    if (params.get('verdict')) state.verdict = params.get('verdict');
    if (params.get('watch') === '1') state.watch = true;
    if (params.get('tab')) state.tab = params.get('tab');
    if (params.get('p')) state.selected = params.get('p');
  }
  function writeHash() {
    var params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.category !== 'all') params.set('category', state.category);
    if (state.verdict !== 'all') params.set('verdict', state.verdict);
    if (state.watch) params.set('watch', '1');
    if (state.tab !== 'pre-token') params.set('tab', state.tab);
    if (state.selected) params.set('p', state.selected);
    var next = params.toString();
    if (next) window.history.replaceState(null, '', '#' + next);
    else window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }

  function filteredRows() {
    var q = state.q.trim().toLowerCase();
    return DATA.rows.filter(function (r) {
      if (state.category !== 'all' && r.category !== state.category) return false;
      if (state.verdict !== 'all' && r.farm.verdict !== state.verdict) return false;
      if (state.watch && DATA.watchlist && DATA.watchlist.indexOf(r.slug) < 0) return false;
      if (!q) return true;
      var hay = [r.name, r.slug, r.category, (r.chains || []).join(' ')].join(' ').toLowerCase();
      return hay.indexOf(q) >= 0;
    });
  }
  function sortedRows(list) {
    var key = state.sort;
    return list.slice().sort(function (a, b) {
      var av, bv;
      if (key === 'name') { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      else if (key === 'tvl') { av = a.tvl_usd; bv = b.tvl_usd; }
      else if (key === 'growth') { av = a.change_7d === null ? -Infinity : a.change_7d; bv = b.change_7d === null ? -Infinity : b.change_7d; }
      else if (key === 'age') { av = a.age_days === null ? Infinity : a.age_days; bv = b.age_days === null ? Infinity : b.age_days; }
      else if (key === 'odds') { av = a.launch_odds; bv = b.launch_odds; }
      else if (key === 'fdv') { av = a.conditional_fdv_usd; bv = b.conditional_fdv_usd; }
      else if (key === 'perhour') { av = a.farm.usd_per_hour; bv = b.farm.usd_per_hour; }
      else { av = a.implied_value_usd; bv = b.implied_value_usd; }
      if (av < bv) return -state.dir;
      if (av > bv) return state.dir;
      return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
    });
  }

  function header() {
    return el('header', { class: 'top' },
      el('div', { class: 'brand' }, el('span', { class: 'mark' }), el('span', { class: 'brand-name', text: 'Token Radar' })),
      el('div', { class: 'top-meta' },
        el('span', { text: 'Snapshot ' + DATA.as_of }),
        el('span', { text: 'DefiLlama protocol index' }),
        el('span', { text: DATA.model_version + ' model' })
      )
    );
  }

  function lede() {
    var t = DATA.totals;
    return el('section', { class: 'lede' },
      el('div', null,
        el('h1', { text: t.watchlist + ' watchlist protocols, ranked by expected launch value' }),
        el('p', { text: 'The watchlist is the floor: every protocol with at least $' + Math.round(DATA.floor_usd / 1e6) + 'M of TVL and no token on record. The early radar adds ' + t.early + ' pre-token protocols below that floor, ranked by the same fitted model. Implied value is a model, not a promise.' })
      ),
      el('div', { class: 'strip' },
        el('div', { class: 'stat' }, el('span', { text: 'Protocols in snapshot' }), el('b', { text: String(t.protocols) }), el('small', { text: t.with_token + ' with a token, ' + t.pre_token + ' without' })),
        el('div', { class: 'stat' }, el('span', { text: 'Board rows' }), el('b', { text: String(t.board) }), el('small', { text: t.watchlist + ' watchlist + ' + t.early + ' early' })),
        el('div', { class: 'stat' }, el('span', { text: 'Snapshot TVL' }), el('b', { text: usd(t.tvl_usd) }), el('small', { text: 'board TVL ' + usd(t.board_tvl_usd) })),
        el('div', { class: 'stat' }, el('span', { text: 'Total implied value' }), el('b', { text: usd(t.total_implied_usd) }), el('small', { text: 'p25 ' + usd(t.implied_p25_usd) + ' - p75 ' + usd(t.implied_p75_usd) })),
        el('div', { class: 'stat' }, el('span', { text: 'Value model, walk-forward' }), el('b', { text: 'RMSLE ' + DATA.value_fit.walk_forward.rmsle.toFixed(2) }), el('small', { text: 'skill ' + DATA.value_fit.walk_forward.skill.toFixed(2) + ' vs median; LOOCV ' + DATA.value_fit.loocv.rmsle.toFixed(2) })),
        el('div', { class: 'stat' }, el('span', { text: 'Farm verdicts' }), el('b', { text: t.farm + ' FARM' }), el('small', { text: t.marginal + ' marginal, ' + t.skip + ' skip' }))
      )
    );
  }

  function controls() {
    var categories = Array.from(new Set(DATA.rows.map(function (r) { return r.category || 'Other'; }))).sort();
    var search = el('input', { type: 'search', id: 'q', placeholder: 'Search name, category or chain', value: state.q, 'aria-label': 'Search the board' });
    search.addEventListener('input', function () { state.q = search.value; writeHash(); render(); });
    var select = el('select', { id: 'category', 'aria-label': 'Filter by category' },
      el('option', { value: 'all', text: 'All categories' }),
      categories.map(function (c) { return el('option', { value: c, text: c, selected: c === state.category ? 'selected' : null }); })
    );
    select.addEventListener('change', function () { state.category = select.value; writeHash(); render(); });
    function pill(label, val) {
      return el('button', { type: 'button', class: 'pill', 'aria-pressed': state.verdict === val ? 'true' : 'false', text: label, onclick: function () { state.verdict = val; writeHash(); render(); } });
    }
    var watchBtn = el('button', { type: 'button', class: 'pill', 'aria-pressed': state.watch ? 'true' : 'false', text: 'Watchlist only', onclick: function () { state.watch = !state.watch; writeHash(); render(); } });
    var pre = el('button', { type: 'button', 'aria-pressed': state.tab === 'pre-token' ? 'true' : 'false', text: 'Pre-token', onclick: function () { state.tab = 'pre-token'; render(); } });
    var live = el('button', { type: 'button', 'aria-pressed': state.tab === 'live' ? 'true' : 'false', text: 'Live token', onclick: function () { state.tab = 'live'; render(); } });
    return el('div', { class: 'controls' },
      el('div', { class: 'field' }, el('label', { for: 'q', text: 'Search' }), search),
      el('div', { class: 'field' }, el('label', { for: 'category', text: 'Category' }), select),
      el('div', { class: 'field' }, el('span', { class: 'sub', text: 'Farm verdict' }), el('div', { class: 'pills' }, pill('All', 'all'), pill('FARM', 'FARM'), pill('Marginal', 'MARGINAL'), pill('Skip', 'SKIP'))),
      el('div', { class: 'field' }, el('span', { class: 'sub', text: 'Scope' }), el('div', { class: 'pills' }, watchBtn)),
      el('div', { class: 'field' }, el('span', { class: 'sub', text: 'Token status' }), el('div', { class: 'tabs' }, pre, live)),
      el('span', { class: 'count', text: state.tab === 'pre-token' ? filteredRows().length + ' rows shown' : DATA.live.length + ' live-token protocols' })
    );
  }

  function sortHeader(key, label) {
    return el('th', { scope: 'col' },
      el('button', { type: 'button', onclick: function () { if (state.sort === key) state.dir = -state.dir; else { state.sort = key; state.dir = -1; } render(); } },
        label,
        state.sort === key ? el('span', { 'aria-hidden': 'true', text: state.dir < 0 ? ' v' : ' ^' }) : null
      )
    );
  }

  function verdictBadge(v) {
    return el('span', { class: 'verdict ' + v.toLowerCase(), text: v });
  }

  function boardTable() {
    var list = sortedRows(filteredRows());
    if (list.length === 0) {
      return el('div', { class: 'panel' }, el('div', { class: 'empty' },
        el('h2', { text: 'No protocol matches that filter' }),
        el('p', { text: 'Clear the search or pick another category to see the board again.' }),
        el('button', { class: 'button', type: 'button', text: 'Clear filters', onclick: function () { state.q = ''; state.category = 'all'; state.verdict = 'all'; state.watch = false; writeHash(); render(); } })
      ));
    }
    var max = list[0].implied_value_usd || 1;
    var tbody = el('tbody');
    list.forEach(function (r) {
      var growth = r.change_7d === null ? 'n/a' : pct(r.change_7d);
      var growthClass = r.change_7d === null ? '' : r.change_7d >= 0 ? 'pos' : 'neg';
      var tr = el('tr', {
        'data-selected': state.selected === r.slug ? 'true' : 'false',
        tabindex: '0',
        onclick: function () { openDetail(r.slug); },
        onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(r.slug); } }
      },
        el('td', { class: 'num', text: String(r.rank) }),
        el('td', null, el('div', { class: 'name' },
          el('b', { text: r.name }),
          el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.slice(0, 2).join(', ') : 'an unnamed chain') })
        )),
        el('td', { class: 'num', text: usd(r.tvl_usd) }),
        el('td', { class: 'num ' + growthClass, text: growth }),
        el('td', { class: 'num', text: ageFmt(r.age_days) }),
        el('td', { class: 'num', text: usd(r.conditional_fdv_usd) }),
        el('td', { class: 'num', text: oddsFmt(r.launch_odds) }),
        el('td', null, el('div', { class: 'implied' },
          el('div', { class: 'bar', 'aria-hidden': 'true' }, el('i', { style: 'width:' + Math.max(3, Math.round((r.implied_value_usd / max) * 100)) + '%' })),
          el('span', { class: 'num', text: usd(r.implied_value_usd) })
        )),
        el('td', { class: 'num', text: (r.farm.usd_per_hour > 0 ? '+' : '') + '$' + r.farm.usd_per_hour.toFixed(2) + '/h' }),
        el('td', null, verdictBadge(r.farm.verdict))
      );
      tbody.appendChild(tr);
    });
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: 'Pre-token protocols, strongest expected launch value first. Select a row for its model, peers and farm plan. $/hour is the model\u2019s expected value per hour (fitted odds x FDV x airdrop share x capital share, minus gas, divided by hours) \u2014 not observed revenue.' }),
          el('thead', null, el('tr', null,
            sortHeader('implied', 'Rank'),
            sortHeader('name', 'Protocol'),
            sortHeader('tvl', 'TVL'),
            sortHeader('growth', '7d'),
            sortHeader('age', 'Age'),
            sortHeader('fdv', 'If it launches'),
            sortHeader('odds', 'Launch odds'),
            sortHeader('implied', 'Implied value'),
            sortHeader('perhour', '$ / hour'),
            el('th', { scope: 'col', text: 'Verdict' })
          )),
          tbody
        )
      )
    );
  }

  function liveTable() {
    var tbody = el('tbody');
    DATA.live.forEach(function (r) {
      tbody.appendChild(el('tr', null,
        el('td', { class: 'num', text: 'live' }),
        el('td', null, el('div', { class: 'name' }, el('b', { text: r.name }), el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.slice(0, 2).join(', ') : 'an unnamed chain') }))),
        el('td', { class: 'num', text: usd(r.tvl_usd) }),
        el('td', { class: 'num ' + (r.change_7d !== null && r.change_7d >= 0 ? 'pos' : 'neg'), text: pct(r.change_7d) }),
        el('td', { class: 'num', text: usd(r.mcap_usd) }),
        el('td', null, el('span', { class: 'status', text: 'already trades' }))
      ));
    });
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: 'Token live: these protocols already trade, so their market cap is a fact, not a model. They are set aside from the pre-token ranking, and they are the peers the value regression was fitted on.' }),
          el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Status' }), el('th', { scope: 'col', text: 'Protocol' }), el('th', { scope: 'col', text: 'TVL' }), el('th', { scope: 'col', text: '7d' }), el('th', { scope: 'col', text: 'Market cap' }), el('th', { scope: 'col', text: 'Token' }))),
          tbody
        )
      )
    );
  }

  function modelPanel() {
    var vf = DATA.value_fit;
    var of = DATA.odds_fit;
    // Watchlist category breakdown, counted from the board rows (all 40 watchlist slugs are board rows).
    var wlCounts = {};
    DATA.rows.forEach(function (r) { if (DATA.watchlist.indexOf(r.slug) >= 0) { var c = r.category || 'Other'; wlCounts[c] = (wlCounts[c] || 0) + 1; } });
    var wlSorted = Object.keys(wlCounts).sort(function (a, b) { return wlCounts[b] - wlCounts[a]; });
    var wlTop3 = wlSorted.slice(0, 3).map(function (c) { return c + ' ' + wlCounts[c]; }).join(', ');
    var wlRest = wlSorted.slice(3).reduce(function (s, c) { return s + wlCounts[c]; }, 0);
    var wlText = 'Watchlist by category (counted from the snapshot): ' + wlTop3 + ', and ' + wlRest + ' others.';
    // Reconciliation with the leaderboard leader's published Steakhouse number, computed from this board.
    var st = DATA.rows.filter(function (r) { return r.slug === 'steakhouse-financial'; })[0];
    var stText = st ? ' Reconciliation with the leaderboard: kiencx\u2019s $98.9M \u201cif it launches\u201d for Steakhouse Financial is the initial float \u2014 ' + (98900000 / st.tvl_usd * 100).toFixed(2) + '% of its ' + usd(st.tvl_usd) + ' TVL. This board\u2019s conditional FDV for the same protocol is ' + usd(st.conditional_fdv_usd) + ' (rank ' + st.rank + '), and its implied value is ' + usd(st.implied_value_usd) + ' at ' + (st.launch_odds * 100).toFixed(1) + '% launch odds. The two \u201cif it launches\u201d numbers differ because one is the float and the other is the full FDV.' : '';
    var coefRows = vf.features.map(function (f, i) {
      return el('tr', null, el('td', { text: f }), el('td', { class: 'num', text: String(vf.coefficients[i]) }));
    });
    return el('section', { class: 'panel model' },
      el('h2', { text: 'How the value is computed' }),
      el('p', { class: 'sub', text: 'One function scores every protocol, fitted from the snapshot itself. The value regression is a Huber-robust, ridge-shrunk (lambda 100 on standardized features) log-log fit of market cap on TVL, seven-day growth, age, chain count and category, trained on the ' + vf.n_train + ' tokens whose market cap the snapshot records (mcap_usd > 0 and tvl_usd > 0). TVL is aggregated across a token\u2019s versions first: versions share a gecko_id or symbol, but the case differs between rows, so the key is lower-cased — in this snapshot ' + (vf.aggregated.length === 0 ? 'no token aggregates' : vf.aggregated.join(' and ') + ' aggregate') + '. Conditional FDV is the predicted market cap divided by the 20% initial float (a stated assumption). Launch odds come from a ridge logistic fit on all ' + of.n + ' protocols. Implied value is odds x FDV. No protocol name is special-cased, and no row is hand-picked.' }),
      el('div', { class: 'formula' }, 'ln(market cap) = ' + vf.coefficients[0] + ' + ' + vf.coefficients[1] + ' x ln(TVL) + ' + vf.coefficients[2] + ' x growth + ' + vf.coefficients[3] + ' x ln(age + 90) + ' + vf.coefficients[4] + ' x ln(chains) + category terms'),
      el('table', { class: 'metrics' },
        el('caption', { class: 'sub', text: 'Fitted coefficients (value regression) and measured validation. A skill ratio below 1 beats predicting the median.' }),
        el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Term' }), el('th', { scope: 'col', text: 'Coefficient' }))),
        el('tbody', null, coefRows)
      ),
      el('table', { class: 'metrics' },
        el('caption', { class: 'sub', text: 'Validation, measured at run time from the snapshot.' }),
        el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Check' }), el('th', { scope: 'col', text: 'Result' }))),
        el('tbody', null,
          el('tr', null, el('td', { text: 'Walk-forward (listed before 2025 vs 2025+)' }), el('td', { class: 'num', text: 'RMSLE ' + vf.walk_forward.rmsle.toFixed(3) + ' - MdAPE ' + (vf.walk_forward.mdape * 100).toFixed(1) + '% - skill ' + vf.walk_forward.skill.toFixed(2) })),
          el('tr', null, el('td', { text: 'Leave-one-out cross-validation' }), el('td', { class: 'num', text: 'RMSLE ' + vf.loocv.rmsle.toFixed(3) + ' - MdAPE ' + (vf.loocv.mdape * 100).toFixed(1) + '% - skill ' + vf.loocv.skill.toFixed(2) })),
          el('tr', null, el('td', { text: 'Odds model: Brier score (in-sample / 5-fold)' }), el('td', { class: 'num', text: of.brier_in_sample + ' / ' + of.brier_cv + ' - AUC ' + of.auc }))
        )
      ),
      el('p', { class: 'sub', text: 'Definitions, counted from the snapshot file itself: \u201cwith a token\u201d means the snapshot records a market cap, a CoinGecko id or a symbol \u2014 ' + DATA.totals.with_token + ' of ' + DATA.totals.protocols + ' protocols; ' + DATA.totals.pre_token + ' record none of the three. ' + wlText }),
      el('p', { class: 'sub', text: 'Farm verdicts follow the launch-odds tier (a stated assumption): FARM at 60% odds or more, MARGINAL at 35% or more, otherwise SKIP \u2014 and only when the best position\u2019s expected net is positive. The cost is gas only ($8 to enter, $2 a week, $3 to exit). No APY or yield is used anywhere: the snapshot carries no yield, fee or points data, so no opportunity cost is charged, and the $/hour column is the model\u2019s expected value per hour, not observed revenue.' + stText }),
      el('p', { class: 'sub', text: 'Assumptions, stated not fitted: 20% of supply circulating at launch; airdrop share 7.5% for DEX and derivatives, 5% elsewhere, 3.5% for RWA; gas about $11 plus $2 a week; no opportunity cost (no yield data in the snapshot). Age is the time since the protocol was first listed; an unlisted age uses the fitted median.' })
    );
  }

  function footer() {
    return el('p', { class: 'foot' },
      el('span', { text: 'Source: ' + DATA.source + '. Snapshot sha256 ' }),
      el('code', { text: DATA.snapshot_sha256.slice(0, 16) + '...' }),
      el('span', { text: '. An implied value is this radar\u2019s claim, never a price and never a promise that a token launches.' })
    );
  }

  var dialog = el('dialog', { class: 'detail', id: 'detail', 'aria-labelledby': 'detail-title' });
  dialog.addEventListener('close', function () { state.selected = null; writeHash(); });

  function nearestPeers(row) {
    var same = DATA.rows.filter(function (r) { return r.slug !== row.slug && r.category === row.category; });
    same.sort(function (a, b) { return Math.abs(Math.log(a.tvl_usd) - Math.log(row.tvl_usd)) - Math.abs(Math.log(b.tvl_usd) - Math.log(row.tvl_usd)); });
    return same.slice(0, 3);
  }

  function hazardSvg(row) {
    var pts = [];
    for (var age = 0; age <= 1825; age += 91) {
      var p = oddsAt(row, age);
      pts.push((age / 1825 * 100).toFixed(1) + ',' + (100 - p * 100).toFixed(1));
    }
    return el('svg', { viewBox: '0 0 100 100', width: '100%', height: '110', 'aria-hidden': 'true', style: 'background:#f7f4ec;border:1px solid #ddd5c2;border-radius:8px' },
      el('polyline', { points: pts.join(' '), fill: 'none', stroke: '#0f766e', 'stroke-width': '2' })
    );
  }

  function openDetail(slug) {
    var row = DATA.rows.filter(function (r) { return r.slug === slug; })[0];
    if (!row) return;
    state.selected = slug;
    writeHash();
    dialog.textContent = '';
    var mcapHat = Math.round(row.conditional_fdv_usd * DATA.assumptions.initial_float);
    // Log-scaled bars: TVL dwarfs the fitted values on a linear scale, so the waterfall
    // shows orders of magnitude, floored so the smallest bar stays visible.
    var logs = [row.tvl_usd, mcapHat, row.conditional_fdv_usd, row.implied_value_usd].map(function (v) { return Math.log10(Math.max(1, v)); });
    var lo = Math.min.apply(null, logs);
    var hi = Math.max.apply(null, logs);
    function wfBar(label, value, dim) {
      var h = hi > lo ? 12 + 76 * ((Math.log10(Math.max(1, value)) - lo) / (hi - lo)) : 50;
      return el('div', { class: 'wf' + (dim ? ' dim' : '') },
        el('i', { style: 'height:' + Math.round(h) + 'px' }),
        el('b', { text: usd(value) }),
        el('span', { text: label })
      );
    }
    var peers = nearestPeers(row);
    var calcOut = el('div', null);
    var calcInput = el('input', { type: 'number', min: '50', max: '100000', step: '50', value: String(row.farm.capital_usd), 'aria-label': 'Position size in US dollars' });
    function drawCalc() {
      var k = Math.max(50, Math.min(100000, Number(calcInput.value) || row.farm.capital_usd));
      var f = farmAt(row, k);
      calcOut.textContent = '';
      calcOut.appendChild(el('span', null,
        el('b', { text: usdFull(f.expected) + ' expected' }),
        el('span', { text: ' against ' + usdFull(f.cost) + ' of costs over ' + holdFor(row.age_days) + ' days, ' }),
        el('b', { text: (f.perHour >= 0 ? '+' : '') + '$' + f.perHour.toFixed(2) + ' an hour' }),
        el('span', { text: ' - verdict ' }),
        verdictBadge(f.verdict)
      ));
    }
    calcInput.addEventListener('input', drawCalc);
    drawCalc();
    dialog.appendChild(el('div', { class: 'detail-head' },
      el('div', null,
        el('span', { class: 'brand-name', text: 'Rank ' + row.rank + ' of ' + DATA.totals.board }),
        el('h2', { id: 'detail-title', text: row.name }),
        el('p', { class: 'sub', text: (row.category || 'Other') + ' on ' + (row.chains && row.chains.length ? row.chains.join(', ') : 'an unnamed chain') + ' - records no token' })
      ),
      el('button', { class: 'close', type: 'button', 'aria-label': 'Close details', text: 'Close', onclick: function () { dialog.close(); } })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Value waterfall' }),
      el('div', { class: 'waterfall' },
        wfBar('TVL', row.tvl_usd, true),
        wfBar('Predicted market cap', mcapHat, false),
        wfBar('If it launches (FDV)', row.conditional_fdv_usd, false),
        wfBar('Implied value (odds x FDV)', row.implied_value_usd, false)
      ),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'TVL' }), el('b', { class: 'num', text: usd(row.tvl_usd) })),
        el('div', null, el('span', { text: 'Seven-day growth' }), el('b', { class: 'num', text: pct(row.change_7d) })),
        el('div', null, el('span', { text: 'Age with no token' }), el('b', { class: 'num', text: ageFmt(row.age_days) })),
        el('div', null, el('span', { text: 'Launch odds' }), el('b', { class: 'num', text: oddsFmt(row.launch_odds) })),
        el('div', null, el('span', { text: 'Expected airdrop pool' }), el('b', { class: 'num', text: usd(row.expected_airdrop_pool_usd) })),
        el('div', null, el('span', { text: 'Implied value' }), el('b', { class: 'num', text: usd(row.implied_value_usd) }))
      ),
      el('p', { class: 'sub', text: 'Model ' + DATA.model_version + ': the regression predicts a market cap from the snapshot figures, the 20% float turns it into an FDV, and the fitted odds turn it into the implied value. This is an estimate, not a price.' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Nearest peers in ' + (row.category || 'the same category') }),
      peers.length === 0
        ? el('p', { class: 'sub', text: 'No other board row shares this category.' })
        : el('table', { class: 'peers' },
            el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Protocol' }), el('th', { scope: 'col', text: 'TVL' }), el('th', { scope: 'col', text: 'If it launches' }), el('th', { scope: 'col', text: 'Implied value' }))),
            el('tbody', null, peers.map(function (p) {
              return el('tr', null,
                el('td', null, el('b', { text: p.name })),
                el('td', { class: 'num', text: usd(p.tvl_usd) }),
                el('td', { class: 'num', text: usd(p.conditional_fdv_usd) }),
                el('td', { class: 'num', text: usd(p.implied_value_usd) })
              );
            }))
          )
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Launch odds over age' }),
      el('p', { class: 'sub', text: 'The fitted odds for this protocol as it ages, holding its TVL, growth and category fixed. Odds rise with age at first — a protocol that survives is more likely to be the kind that launches — and the fitted curve turns down only past about eleven years.' }),
      hazardSvg(row)
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Farm plan' }),
      el('div', { class: 'calc' },
        el('label', { for: 'farm-capital', text: 'Position size $' }),
        calcInput,
        calcOut
      ),
      el('ol', { class: 'steps' }, row.farm_steps.map(function (s) {
        var at = s.indexOf(' Cost: ');
        var action = at >= 0 ? s.slice(0, at) : s;
        var tail = at >= 0 ? s.slice(at + 1) : '';
        return el('li', null, el('b', { text: action }), el('span', { class: 'cost', text: tail }));
      }))
    ));
    if (typeof dialog.showModal === 'function') {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }
  }

  function render() {
    app.textContent = '';
    var shell = el('div', { class: 'shell' });
    shell.appendChild(header());
    shell.appendChild(lede());
    shell.appendChild(controls());
    shell.appendChild(state.tab === 'pre-token' ? boardTable() : liveTable());
    if (state.tab === 'pre-token') shell.appendChild(modelPanel());
    shell.appendChild(footer());
    app.appendChild(shell);
    app.setAttribute('aria-busy', 'false');
    if (!dialog.isConnected) app.appendChild(dialog);
    if (state.selected) openDetail(state.selected);
  }

  readHash();
  render();
  window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && dialog.open) dialog.close();
  });
})();
