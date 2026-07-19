#!/usr/bin/env python3
"""Build a standalone HTML evaluation viewer from a finished run folder.
Usage: python tools/build_viewer.py <run_folder>   ->  writes <run>_viewer.html next to it.
Offline, no dependencies at view time. Requires networkx for layout at build time."""
import csv, glob, os, json, sys, re
import networkx as nx

# scripts/ holds the shared metric + naming modules used across the repo
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import opinion_metrics  # shared B/D/P definitions

def _clean(t):
    t = str(t or "").strip()
    t = re.sub(r'^FINAL_RATING:\s*-?\d+\s*(?:\n|\\n|\s)*(?:TWEET:|EXPLANATION:)\s*', '', t, flags=re.I)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def build_data(run_dir):
    one = lambda p: (glob.glob(os.path.join(run_dir, p)) or [None])[0]
    rows = list(csv.reader(open(one("*network_opinion_change*.csv"), encoding="utf-8")))
    names = rows[0][1:]; steps = [r[0] for r in rows[1:]]
    series = {n: [] for n in names}
    for r in rows[1:]:
        for i, n in enumerate(names):
            try: series[n].append(int(r[i+1]))
            except: series[n].append(0)
    efiles = sorted(glob.glob(os.path.join(run_dir, "*step_*edges.csv")))
    edges = []
    if efiles:
        for r in csv.DictReader(open(efiles[len(efiles)//2], encoding="utf-8")):
            a, b = r.get("i_name"), r.get("j_name")
            if a in series and b in series: edges.append((a, b))
    G = nx.Graph(); G.add_nodes_from(names); G.add_edges_from(edges)
    deg = dict(G.degree()); hubs = set(sorted(deg, key=deg.get, reverse=True)[:2])
    # structural communities (greedy modularity, Clauset-Newman-Moore) for the viewer's community view
    comm_of = {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        for ci, grp in enumerate(greedy_modularity_communities(G)):
            for n_ in grp:
                comm_of[n_] = ci
    except Exception:
        comm_of = {}
    pos = nx.spring_layout(G, seed=3, k=0.9, iterations=200) if edges else {n:(0,0) for n in names}
    written = {n: [] for n in names}; read = {n: [] for n in names}; speaks = []
    meta = {}
    def _flag(v):
        s = str(v or "").strip().lower()
        return 0 if s in ("", "0", "0.0", "nan", "false", "none") else 1
    ic = one("*network_interactions*.csv")
    if ic:
        for r in csv.DictReader(open(ic, encoding="utf-8")):
            if not meta:
                meta = {"world": (r.get("World") or "").strip(), "rag": (r.get("RAG Content Mode") or "").strip(), "factpack": (r.get("Fact Pack Mode") or "").strip(), "version": (r.get("Version Set") or "").strip()}
            st = r.get("Time Step"); spk = r.get("Agent_J Name"); lis = r.get("Agent_I Name")
            tw = _clean(r.get("Agent_J Tweet")); resp = _clean(r.get("Agent_I Response"))
            try: d = int(r.get("Agent_I Delta-Belief") or 0)
            except: d = 0
            if spk in written and tw:
                written[spk].append({"step": st, "text": tw, "fb": _flag(r.get("Agent_J Step2 Fallback Used")), "warn": (r.get("Agent_J Step2 Warning Tags") or "").strip()})
            if lis in read:
                read[lis].append({"step": st, "from": spk, "text": tw, "dir": d, "resp": resp,
                    "pre": (r.get("Agent_I Pre-Belief") or "").strip(), "post": (r.get("Agent_I Post-Belief") or "").strip(),
                    "allowed": (r.get("Agent_I Allowed Ratings") or "").strip(),
                    "rep": _flag(r.get("Agent_I Hard Repair")), "soft": _flag(r.get("Agent_I Soft Cleanup"))})
            if spk and lis: speaks.append({"step": st, "from": spk, "to": lis})
    changes = {n: [] for n in names}
    for n in names:
        prev = series[n][0]
        for i in range(1, len(series[n])):
            if series[n][i] != prev:
                changes[n].append({"step": steps[i], "from": prev, "to": series[n][i], "dir": 1 if series[n][i] > prev else -1}); prev = series[n][i]
    def bdp(v):
        B, D, P = opinion_metrics.bdp(v)
        return round(B,3), round(D,3), round(P,3)
    bdpser=[dict(zip(("step","B","D","P"),(steps[i],)+bdp([series[n][i] for n in names]))) for i in range(len(steps))]
    cfg = {}
    mj = one("metrics_*.json")
    if mj:
        try: cfg = json.load(open(mj, encoding="utf-8")).get("config", {})
        except: cfg = {}
    for _k, _v in meta.items():
        if not cfg.get(_k): cfg[_k] = _v
    # personas: look up the version's list_agent_descriptions.csv (same source the run used)
    personas = {}
    version_set = (meta.get("version") or "").strip()
    if version_set:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hits = glob.glob(os.path.join(repo, "prompts", "**", version_set), recursive=True)
        vdir = next((h for h in sorted(hits) if os.path.isdir(h)), None)
        pcsv = os.path.join(vdir, "list_agent_descriptions.csv") if vdir else None
        if pcsv and os.path.exists(pcsv):
            demo_cols = ["political_leaning", "age", "gender", "ethnicity", "education", "occupation",
                         "epistemic_profile", "institutional_trust", "uncertainty_tolerance",
                         "evidence_style", "official_narrative_suspicion", "openness_to_update"]
            try:
                for row in csv.DictReader(open(pcsv, encoding="utf-8")):
                    nm = (row.get("agent_name") or "").strip()
                    if nm not in series: continue
                    traits = {k: row[k].strip() for k in demo_cols if (row.get(k) or "").strip()}
                    card = (row.get("persona") or "").strip()
                    personas[nm] = {"card": card[:600], "traits": traits}
            except Exception:
                personas = {}
    adj = {n: set() for n in names}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    ids = {n: f"A{i}" for i, n in enumerate(names)}
    agents = [{"id": ids[n], "name": n, "op": series[n][-1], "hub": n in hubs,
               "x": round(pos[n][0],3), "y": round(pos[n][1],3), "series": series[n],
               "degree": len(adj[n]), "neighbors": sorted(ids[m] for m in adj[n]),
               "pers": personas.get(n) or None,
               "comm": comm_of.get(n, 0),
               "written": written[n], "read": read[n], "changes": changes[n]} for n in names]
    return {"run": os.path.basename(run_dir), "config": cfg, "steps": steps, "agents": agents,
            "edges": [[ids[a], ids[b]] for a, b in edges], "interactions": speaks, "bdp": bdpser}

TEMPLATE = r'''<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Run viewer</title><style>
html,body{margin:0;height:100%;background:#06080f;color:#c8d2e6;font-family:-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
.wrap{display:flex;height:100vh;position:relative}.scene{flex:1;position:relative}svg{width:100%;height:100%}
.hd{position:absolute;top:12px;left:16px;font-size:13px}.hd b{color:#eaf1ff}
.hud{position:absolute;top:44px;left:16px;background:rgba(10,14,26,.7);border:1px solid #1e2740;border-radius:9px;padding:9px 12px;font-size:12px;line-height:1.7}
.hud .k{color:#6f7c99;display:inline-block;width:70px}.hud b{color:#eaf1ff}
.hud .met{display:flex;gap:14px;margin-top:6px;padding-top:6px;border-top:1px solid #1e2740}.hud .met b{display:block;font-size:15px}
.legend{position:absolute;bottom:56px;right:18px;font-size:11px;color:#8b96ad;display:flex;gap:8px;align-items:center}
.bar{width:140px;height:9px;border-radius:5px;background:linear-gradient(90deg,#d92b2b 0 20%,#f59331 20% 40%,#8f96ad 40% 60%,#39c0e0 60% 80%,#2f6bff 80% 100%);position:relative}
#meanmk{position:absolute;top:-4px;width:3px;height:16px;background:#fff;border-radius:2px;box-shadow:0 0 5px #fff}
.scrub{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);display:flex;gap:12px;align-items:center;background:rgba(10,14,26,.85);border:1px solid #1e2740;border-radius:10px;padding:8px 14px}
.scrub input{width:340px}.scrub .lbl{font-size:12px;color:#b9c4dc;min-width:120px}.pbtn{background:#1e2b4d;border:0;color:#fff;width:28px;height:28px;border-radius:50%;cursor:pointer}
.inter{position:absolute;top:14px;left:50%;transform:translateX(-50%);font-size:12px;color:#9fb0d0;background:rgba(10,14,26,.6);padding:3px 12px;border-radius:8px}
.panel{width:0;transition:width .25s;background:rgba(9,13,24,.97);border-left:1px solid #1e2740;overflow:hidden}.panel.open{width:380px}
.pin{padding:18px;width:380px;box-sizing:border-box;height:100%;overflow-y:auto;position:relative}
.pin h2{margin:0;font-size:16px;color:#eaf1ff;display:flex;gap:8px;align-items:center}.dot{width:12px;height:12px;border-radius:50%}
.sub{color:#7d88a3;font-size:12px;margin:4px 0 12px}.sec{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#6f7c99;margin:14px 0 6px}
.msg{border:1px solid #1b2440;border-radius:8px;margin-bottom:6px;overflow:hidden;background:#111a30}
.msg .row{display:flex;gap:8px;padding:8px 10px;cursor:pointer;font-size:12.5px}.msg .row:hover{background:#16203a}
.msg .arw{color:#5b678a;font-size:11px;transition:transform .18s}.msg.open .arw{transform:rotate(90deg)}.msg .lab{flex:1;color:#c8d2e6}.msg .stp{color:#5b678a;font-size:11px}
.msg .body{max-height:0;overflow:hidden;transition:max-height .22s;padding:0 10px;color:#9aa7c2;font-size:12.5px;line-height:1.5}.msg.open .body{max-height:260px;padding:0 10px 10px}
.chg{display:flex;gap:8px;font-size:12.5px;padding:6px 10px;border:1px solid #1b2440;border-radius:8px;margin-bottom:6px;background:#111a30}.up{color:#5fd0ff}.down{color:#ff6a6a}
.close{position:absolute;top:12px;right:14px;cursor:pointer;color:#7684a0;font-size:20px}
.morebtn{width:100%;text-align:left;background:#111a30;border:1px solid #1b2440;color:#c8d2e6;padding:9px 11px;border-radius:8px;cursor:pointer;font-size:12.5px;margin-top:12px}.morebtn:hover{background:#16203a}
.more{display:none;border:1px solid #1b2440;border-top:0;border-radius:0 0 8px 8px;padding:10px 11px;font-size:12.5px;line-height:1.85;color:#9aa7c2}.more .k2{color:#6f7c99;display:inline-block;width:110px}.more b{color:#eaf1ff}
.msg.eup{border-left:3px solid #5fd0ff}.msg.edn{border-left:3px solid #ff6a6a}
.jump{cursor:pointer;text-decoration:underline dotted #47506e}.jump:hover{color:#9fb0d0}
.stepchg{font-size:11px;color:#8b96ad;margin-left:4px;min-width:70px}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0 8px}.chip{font-size:11px;color:#8b96ad;background:#0e1526;border:1px solid #1b2440;border-radius:12px;padding:3px 9px;cursor:pointer}.chip.on{background:#1e2b4d;color:#eaf1ff;border-color:#2d3f6b}.chip:hover{color:#eaf1ff}
.slidewrap{position:relative;width:340px;height:22px;display:flex;align-items:center}.slidewrap input{width:340px;position:relative;z-index:1}
.ticks{position:absolute;left:0;right:0;top:0;height:7px;pointer-events:none}.ticks i{position:absolute;top:0;width:2px;transform:translateX(-1px);border-radius:1px}
.tgl{position:absolute;top:44px;right:18px;background:rgba(10,14,26,.7);border:1px solid #1e2740;color:#b9c4dc;font-size:11.5px;border-radius:8px;padding:6px 10px;cursor:pointer}.tgl.on{background:#1e2b4d;color:#fff}
.distbox{position:absolute;top:262px;right:18px;width:214px;background:rgba(10,14,26,.93);border:1px solid #1e2740;border-radius:10px;padding:12px;display:none}.distbox.on{display:block}.trajbox{width:322px}
.distbox .db{display:flex;align-items:flex-end;gap:8px;height:96px}.distbox .col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}.distbox .col b{width:100%;border-radius:3px 3px 0 0;min-height:2px}.distbox .col span{font-size:10px;color:#8b96ad;margin-top:4px}.distbox .cnt{font-size:10px;color:#eaf1ff;height:12px}
.tip{position:absolute;pointer-events:none;background:rgba(6,10,20,.96);border:1px solid #2d3f6b;border-radius:6px;padding:4px 8px;font-size:11.5px;color:#eaf1ff;display:none;z-index:5}
.srch{position:absolute;top:12px;right:18px;width:170px;background:#0e1526;border:1px solid #1e2740;border-radius:8px;color:#eaf1ff;font-size:12px;padding:5px 9px;outline:none}.srch::placeholder{color:#5b678a}
.tgl.f2{top:116px}.tgl.f3{top:152px}
.perst{font-size:12.5px;line-height:1.8;color:#9aa7c2}.perst .k2{color:#6f7c99;display:inline-block;width:150px;vertical-align:top}.perst b{color:#eaf1ff}
.perscard{border:1px solid #1b2440;border-radius:8px;background:#111a30;padding:8px 10px;font-size:12px;line-height:1.5;color:#c8d2e6;white-space:pre-wrap;margin-top:6px}
.ex{margin-bottom:5px}.exlab{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.4px;color:#5b678a;margin-right:6px;background:#0e1526;border:1px solid #1b2440;border-radius:4px;padding:1px 5px}.exmeta{font-size:11px;color:#7d88a3;margin-top:4px}.flag{font-size:10px;color:#ffcf7a;background:rgba(120,90,20,.25);border:1px solid #5a4a20;border-radius:4px;padding:1px 5px;margin-left:4px}
.sec.sechead{cursor:pointer;display:flex;align-items:center;gap:7px;user-select:none}.sec.sechead:hover{color:#9fb0d0}.sec .cav{transition:transform .18s;font-size:9px;color:#5b678a}.sec.collapsed .cav{transform:rotate(-90deg)}.secbody{overflow:hidden}.sec.collapsed + .secbody{display:none}
.more2{font-size:12.5px;line-height:1.85;color:#9aa7c2;padding:2px 2px 6px}.more2 .k2{color:#6f7c99;display:inline-block;width:110px}.more2 b{color:#eaf1ff}
.drawer{position:absolute;top:50%;transform:translateY(-50%);right:0;transition:right .25s;z-index:6;background:#1e2b4d;color:#fff;border:1px solid #2d3f6b;border-right:0;border-radius:9px 0 0 9px;width:22px;height:64px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:14px}.drawer:hover{background:#26356a}
</style></head><body><div class="wrap"><div class="scene">
<div class="hd"><b id="runname"></b></div><div class="hud" id="hud"></div><div class="inter" id="interlbl"></div><button class="tgl" id="distbtn" style="top:44px">&#9638; Distribution</button><input class="srch" id="srch" placeholder="&#128269; find agent (Enter)" list="agnames"/><datalist id="agnames"></datalist><div class="distbox" id="dist"></div><button class="tgl" id="trajbtn" style="top:80px">&#9683; Trajectories</button><button class="tgl f2" id="fltchg">&#9650;&#9660; Changed only</button><button class="tgl f3" id="flthub">&#9733; Hubs only</button><button class="tgl" id="commbtn" style="top:188px">&#9673; Communities</button><button class="tgl" id="pngbtn" style="top:224px">&#8681; PNG</button><div class="distbox trajbox" id="traj"></div><div class="tip" id="tip"></div>
<div class="legend"><span>reject</span><span class="bar"><i id="meanmk"></i></span><span>accept</span></div>
<div class="scrub"><button class="pbtn" id="play">&#9654;</button><span class="lbl" id="slbl"></span><span class="slidewrap"><span class="ticks" id="ticks"></span><input type="range" id="step" min="0" value="0"/></span><span class="stepchg" id="stepchg"></span></div>
<svg id="svg" viewBox="0 0 1000 640" preserveAspectRatio="xMidYMid meet"></svg></div>
<div class="panel" id="panel"><div class="pin" id="pin"></div></div><div class="drawer" id="drawer">&#8249;</div></div>
<script>const DATA=__DATA__;
(function(){
 const svg=document.getElementById('svg'),NS='http://www.w3.org/2000/svg';
 const A=DATA.agents,by={};A.forEach(a=>by[a.id]=a);
 const xs=A.map(a=>a.x),ys=A.map(a=>a.y),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const SX=v=>90+((v-x0)/((x1-x0)||1))*820, SY=v=>70+((v-y0)/((y1-y0)||1))*500;
 A.forEach(a=>{a.px=SX(a.x);a.py=SY(a.y);});
 const NS_=DATA.steps.length;
 const OPCOL={'-2':'#d92b2b','-1':'#f59331','0':'#8f96ad','1':'#39c0e0','2':'#2f6bff'};
 function col(op){return OPCOL[Math.max(-2,Math.min(2,Math.round(op)))]||'#8f96ad';}
 // edges
 DATA.edges.forEach(e=>{const a=by[e[0]],b=by[e[1]];if(!a||!b)return;const l=document.createElementNS(NS,'line');
  l.setAttribute('x1',a.px);l.setAttribute('y1',a.py);l.setAttribute('x2',b.px);l.setAttribute('y2',b.py);l.setAttribute('stroke','#26324f');l.setAttribute('stroke-width','1.2');svg.appendChild(l);});
 const arrow=document.createElementNS(NS,'line');arrow.setAttribute('stroke','#ffe08a');arrow.setAttribute('stroke-width','2.4');arrow.setAttribute('opacity','0');svg.appendChild(arrow);
 function halo(cl){const c=document.createElementNS(NS,'circle');c.setAttribute('fill','none');c.setAttribute('stroke',cl);c.setAttribute('stroke-width','3');c.setAttribute('opacity','0');svg.appendChild(c);return c;}
 function slab(cl,tx){const e=document.createElementNS(NS,'text');e.setAttribute('fill',cl);e.setAttribute('font-size','11');e.setAttribute('font-weight','bold');e.setAttribute('text-anchor','middle');e.setAttribute('opacity','0');e.textContent=tx;svg.appendChild(e);return e;}
 const spkHalo=halo('#ffd166'),lisHalo=halo('#ff8a3d'),spkLab=slab('#ffd166','S'),lisLab=slab('#ff8a3d','L');
 // nodes
 const nodeEls={};
 A.forEach(a=>{const g=document.createElementNS(NS,'g');g.style.cursor='pointer';
  const c=document.createElementNS(NS,'circle');c.setAttribute('cx',a.px);c.setAttribute('cy',a.py);c.setAttribute('r',a.hub?15:9);c.setAttribute('stroke','#0a0f1c');c.setAttribute('stroke-width','2');g.appendChild(c);
  const t=document.createElementNS(NS,'text');t.setAttribute('x',a.px);t.setAttribute('y',a.py-(a.hub?20:14));t.setAttribute('text-anchor','middle');t.setAttribute('fill','#aeb9d2');t.setAttribute('font-size','10');t.textContent=(a.hub?'★ ':'')+a.name;g.appendChild(t);
  g.addEventListener('click',()=>open(a));
  g.addEventListener('mousemove',ev=>{const tp=document.getElementById('tip');const op=opAt(a,cur);tp.style.display='block';tp.style.left=(ev.clientX+12)+'px';tp.style.top=(ev.clientY+12)+'px';tp.innerHTML=`<b>${a.name}</b> · opinion ${op>=0?'+':''}${op}`;});
  g.addEventListener('mouseleave',()=>{document.getElementById('tip').style.display='none';});
  svg.appendChild(g);nodeEls[a.id]=c;});
 let selected=null,filterMode='all';
 function passFilter(a){return filterMode==='all'||(filterMode==='chg'&&a.changes.length>0)||(filterMode==='hub'&&a.hub);}
 function highlight(){A.forEach(x=>{const vis=passFilter(x);const on=vis&&(!selected||x.id===selected.id||(selected.neighbors||[]).indexOf(x.id)>=0);nodeEls[x.id].setAttribute('opacity',on?'1':(vis?'0.16':'0.05'));});}
 function setFilter(m){filterMode=(filterMode===m?'all':m);
  document.getElementById('fltchg').classList.toggle('on',filterMode==='chg');
  document.getElementById('flthub').classList.toggle('on',filterMode==='hub');
  highlight();}
 document.getElementById('fltchg').addEventListener('click',()=>setFilter('chg'));
 document.getElementById('flthub').addEventListener('click',()=>setFilter('hub'));
 // search: type a name (datalist suggests), Enter selects & opens the agent
 {const s=document.getElementById('srch'),dl=document.getElementById('agnames');
  A.forEach(a=>{const o=document.createElement('option');o.value=a.name;dl.appendChild(o);});
  function go(){const q=(s.value||'').trim().toLowerCase();if(!q)return;
   const a=A.find(x=>x.name.toLowerCase()===q)||A.find(x=>x.name.toLowerCase().indexOf(q)>=0);
   if(a){open(a);}}
  s.addEventListener('change',go);
  s.addEventListener('keydown',e=>{if(e.key==='Enter'){go();e.stopPropagation();}});}
 function ring(i){let up=0,dn=0;A.forEach(a=>{let d=0;if(i>0){const c=Math.round(opAt(a,i)),p=Math.round(opAt(a,i-1));d=c>p?1:c<p?-1:0;}if(d>0)up++;else if(d<0)dn++;nodeEls[a.id].setAttribute('stroke',d>0?'#5fd0ff':d<0?'#ff6a6a':'#0a0f1c');nodeEls[a.id].setAttribute('stroke-width',d?'3.5':'2');});
  const el=document.getElementById('stepchg');if(el)el.innerHTML=(up+dn)?('&#916; <span class="up">&#9650;'+up+'</span> <span class="down">&#9660;'+dn+'</span>'):'<span style="color:#4b5674">no change</span>';}
 function mean(x){return x.reduce((s,v)=>s+v,0)/x.length;}
 function opAt(a,i){return a.series[Math.min(i,a.series.length-1)];}
 const COMMPAL=['#4dabf7','#ffa94d','#69db7c','#e599f7','#ff8787','#66d9e8','#ffd43b','#b197fc'];
 let commOn=false;
 function recolor(i){A.forEach(a=>{const f=commOn?COMMPAL[(a.comm||0)%COMMPAL.length]:col(opAt(a,i));nodeEls[a.id].setAttribute('fill',f);});}
 document.getElementById('commbtn').addEventListener('click',function(){commOn=!commOn;this.classList.toggle('on',commOn);recolor(cur);});
 // PNG export: serialize the SVG, paint it on a canvas, download as PNG (client-side only)
 document.getElementById('pngbtn').addEventListener('click',function(){
  try{
   const ser=new XMLSerializer().serializeToString(svg);
   const blob=new Blob([ser],{type:'image/svg+xml;charset=utf-8'});
   const url=URL.createObjectURL(blob);
   const img=new Image();
   img.onload=function(){
    const cv=document.createElement('canvas');cv.width=2000;cv.height=1280;
    const cx=cv.getContext('2d');cx.fillStyle='#06080f';cx.fillRect(0,0,cv.width,cv.height);
    cx.drawImage(img,0,0,cv.width,cv.height);
    URL.revokeObjectURL(url);
    const a=document.createElement('a');a.download=(DATA.run||'network')+'_step'+DATA.steps[Math.min(cur,DATA.steps.length-1)]+'.png';
    a.href=cv.toDataURL('image/png');a.click();
   };
   img.src=url;
  }catch(e){alert('PNG export failed: '+e.message);}
 });
 function bdpAt(i){return DATA.bdp[Math.min(i,DATA.bdp.length-1)];}
 const hud=document.getElementById('hud'),cfg=DATA.config||{};
 document.getElementById('runname').textContent=DATA.run;
 function renderHUD(i){const m=bdpAt(i);
  hud.innerHTML=`<div><span class="k">Model</span><b>${cfg.model_name||'?'}</b></div><div><span class="k">Version</span><b>${cfg.version_set||'?'}</b></div><div><span class="k">Strictness</span><b>${cfg.validation_strictness||'?'}</b></div><div><span class="k">World</span><b>${cfg.world||'?'}</b></div><div><span class="k">RAG</span><b>${cfg.rag||'?'}</b></div><div><span class="k">Step</span><b>${DATA.steps[i]} / ${DATA.steps[DATA.steps.length-1]}</b></div><div class="met"><div>B<b>${(+m.B).toFixed(2)}</b></div><div>D<b>${(+m.D).toFixed(2)}</b></div><div>P<b>${(+m.P).toFixed(2)}</b></div></div>`;}
 function hideSL(){[spkHalo,lisHalo,spkLab,lisLab].forEach(e=>e.setAttribute('opacity','0'));}
 function updateArrow(i){const st=DATA.steps[i];const it=DATA.interactions.find(x=>x.step==st);
  if(!it){arrow.setAttribute('opacity','0');hideSL();document.getElementById('interlbl').textContent='';return;}
  const fa=A.find(a=>a.name===it.from),la=A.find(a=>a.name===it.to);if(!fa||!la){arrow.setAttribute('opacity','0');hideSL();return;}
  arrow.setAttribute('x1',fa.px);arrow.setAttribute('y1',fa.py);arrow.setAttribute('x2',la.px);arrow.setAttribute('y2',la.py);arrow.setAttribute('opacity','.85');
  const rf=fa.hub?15:9,rl=la.hub?15:9;
  spkHalo.setAttribute('cx',fa.px);spkHalo.setAttribute('cy',fa.py);spkHalo.setAttribute('r',rf+5);spkHalo.setAttribute('opacity','.95');
  lisHalo.setAttribute('cx',la.px);lisHalo.setAttribute('cy',la.py);lisHalo.setAttribute('r',rl+5);lisHalo.setAttribute('opacity','.95');
  spkLab.setAttribute('x',fa.px+rf+9);spkLab.setAttribute('y',fa.py-rf-1);spkLab.setAttribute('opacity','1');
  lisLab.setAttribute('x',la.px+rl+9);lisLab.setAttribute('y',la.py-rl-1);lisLab.setAttribute('opacity','1');
  document.getElementById('interlbl').innerHTML=`step ${st} · <b style="color:#ffd166">S</b> ${it.from} &#8594; <b style="color:#ff8a3d">L</b> ${it.to}`;}
 let cur=0;const inp=document.getElementById('step');inp.max=NS_-1;
 function setStep(i){cur=i;document.getElementById('slbl').textContent='Step '+DATA.steps[i]+' / '+DATA.steps[DATA.steps.length-1];renderHUD(i);recolor(i);ring(i);updateArrow(i);var _m=bdpAt(i),mk=document.getElementById('meanmk');if(mk)mk.style.left=((+_m.B+2)/4*100)+'%';if(distOn)renderDist(i);if(trajOn)updateTrajMk(i);highlight();}
 inp.addEventListener('input',e=>setStep(+e.target.value));
 let playing=false,loop=false,timer=null,pressT=null,longP=false;const pb=document.getElementById('play');
 function stopPlay(){playing=false;clearInterval(timer);pb.innerHTML='&#9654;';}
 function startPlay(lp){loop=lp;playing=true;pb.innerHTML=lp?'&#9632;':'&#10074;&#10074;';clearInterval(timer);
  timer=setInterval(()=>{let n=cur+1;if(n>=NS_){if(loop){n=0;}else{stopPlay();return;}}inp.value=n;setStep(n);},350);}
 pb.title='Click: play once  ·  Hold: loop';
 pb.addEventListener('mousedown',()=>{longP=false;pressT=setTimeout(()=>{longP=true;startPlay(true);},400);});
 pb.addEventListener('mouseup',()=>{clearTimeout(pressT);if(longP)return;if(playing)stopPlay();else startPlay(false);});
 pb.addEventListener('mouseleave',()=>clearTimeout(pressT));
 const panel=document.getElementById('panel'),pin=document.getElementById('pin');
 const drawer=document.getElementById('drawer');
 function updateDrawer(){const op=panel.classList.contains('open');drawer.style.right=op?'380px':'0px';drawer.innerHTML=op?'&#8250;':'&#8249;';drawer.title=op?'Hide panel':'Show panel';}
 drawer.addEventListener('click',()=>{if(panel.classList.contains('open')){panel.classList.remove('open');}else if(selected){panel.classList.add('open');}updateDrawer();highlight();});
 window.closePanel=function(){panel.classList.remove('open');selected=null;highlight();renderTicks(null);updateDrawer();};
 function spark(a){const W=340,H=44,n=a.series.length;let p='';a.series.forEach((v,i)=>{const x=i/(n-1)*W,y=H-((v+2)/4)*H;p+=(i?' L':'M')+x.toFixed(1)+' '+y.toFixed(1);});
  let d='';a.changes.forEach(c=>{let i=DATA.steps.indexOf(c.step);if(i<0)i=0;const x=i/(n-1)*W,y=H-((c.to+2)/4)*H;d+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${c.dir>0?'#5fd0ff':'#ff6a6a'}"><title>step ${c.step}: ${c.from}→${c.to}</title></circle>`;});
  return `<svg width="${W}" height="${H}" style="margin:6px 0 12px"><line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}" stroke="#1b2440"/><path d="${p}" fill="none" stroke="#c8d2e6" stroke-width="1.6"/>${d}</svg>`;}
 function stanceAt(ag,step){let si=DATA.steps.indexOf(step);if(si<0)si=DATA.steps.indexOf(+step);if(si<0)si=0;return ag.series[Math.min(si,ag.series.length-1)];}
 function sideOf(a,m){const spk=A.find(x=>x.name===m.from);if(!spk)return 'r-neu';const ss=stanceAt(spk,m.step),ls=stanceAt(a,m.step);if((ss>0&&ls>0)||(ss<0&&ls<0))return 'r-same';if((ss>0&&ls<0)||(ss<0&&ls>0))return 'r-opp';return 'r-neu';}
 function flagsOf(arr){return arr.filter(Boolean).map(x=>`<span class="flag">${x}</span>`).join('');}
 function rows(list,kind,ag){if(!list.length)return '<div class="sub">none</div>';let h='';list.slice().reverse().forEach(m=>{
  if(kind==='w'){const op=stanceAt(ag,m.step);const st=(op>=0?'+':'')+op;const cc=op>0?'w-for':op<0?'w-against':'w-neu';const fl=(m.fb||m.warn)?'w-flag':'';const fb=flagsOf([m.fb?'fallback':'',m.warn?('warn: '+m.warn):'']);h+=`<div class="msg ${cc} ${fl}"><div class="row"><span class="arw">&#8250;</span><span class="lab">wrote a post · stance <b>${st}</b>${fl?' <span class="flag">flagged</span>':''}</span><span class="stp jump" data-step="${m.step}">step ${m.step} &#8615;</span></div><div class="body">${m.text||'(empty)'}${fb?('<div class="exmeta">'+fb+'</div>'):''}</div></div>`;}
  else{const e=m.dir>0?'moved +1':m.dir<0?'moved -1':'no change';const cl=m.dir>0?'eup':m.dir<0?'edn':'';const fl=(m.rep||m.soft)?'r-flag':'';const cc=(m.dir?'r-changed':'r-none')+' '+sideOf(ag,m)+' '+fl;const fb=flagsOf([m.rep?'repaired':'',m.soft?'cleaned':'']);
   h+=`<div class="msg ${cl} ${cc}"><div class="row"><span class="arw">&#8250;</span><span class="lab">read from ${m.from} · <span class="${m.dir>0?'up':m.dir<0?'down':''}">${e}</span>${fl?' <span class="flag">flagged</span>':''}</span><span class="stp jump" data-step="${m.step}">step ${m.step} &#8615;</span></div><div class="body"><div class="ex"><span class="exlab">read</span>"${m.text||'(empty)'}"</div><div class="ex"><span class="exlab">reply</span>${m.resp||'&#8212;'}</div><div class="exmeta">belief ${m.pre} &#8594; ${m.post}${m.allowed?(' · allowed '+m.allowed):''}${fb?(' · '+fb):''}</div></div></div>`;}});return h;}
 function chip(cls,lab,on){return `<span class="chip${on?' on':''}" data-cls="${cls}">${lab}</span>`;}
 function chipsW(a){let fo=0,ne=0,ag=0,fl=0;a.written.forEach(m=>{const s=stanceAt(a,m.step);if(s>0)fo++;else if(s<0)ag++;else ne++;if(m.fb||m.warn)fl++;});return `<div class="chips">${chip('','all '+a.written.length,1)}${chip('w-for','\u25B2 for '+fo)}${chip('w-neu','\u25E6 mixed '+ne)}${chip('w-against','\u25BC against '+ag)}${fl?chip('w-flag','\u26A0 flagged '+fl):''}</div>`;}
 function chipsR(a){let ch=0,no=0,sm=0,op=0,fl=0;a.read.forEach(m=>{if(m.dir)ch++;else no++;const sd=sideOf(a,m);if(sd==='r-same')sm++;else if(sd==='r-opp')op++;if(m.rep||m.soft)fl++;});return `<div class="chips">${chip('','all '+a.read.length,1)}${chip('r-changed','moved me '+ch)}${chip('r-none','no effect '+no)}${chip('r-same','same-side '+sm)}${chip('r-opp','opposing '+op)}${fl?chip('r-flag','\u26A0 flagged '+fl):''}</div>`;}
 function wireChips(box){const list=box.nextElementSibling;box.querySelectorAll('.chip').forEach(ch=>ch.addEventListener('click',()=>{box.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));ch.classList.add('on');const cls=ch.getAttribute('data-cls');list.querySelectorAll('.msg').forEach(m=>{m.style.display=(!cls||m.classList.contains(cls))?'':'none';});}));}
 const sysChg=DATA.steps.map((s,i)=>{if(i===0)return 0;let c=0;A.forEach(a=>{if(Math.round(opAt(a,i))!==Math.round(opAt(a,i-1)))c++;});return c;});
 const maxChg=Math.max(1,...sysChg);const ticksEl=document.getElementById('ticks');
 function renderTicks(sel){let h='';sysChg.forEach((c,i)=>{if(!c)return;const L=i/(NS_-1)*100,ht=3+(c/maxChg)*4;h+=`<i style="left:${L}%;height:${ht}px;background:#37476e"></i>`;});
  if(sel){sel.changes.forEach(cg=>{let i=DATA.steps.indexOf(cg.step);if(i<0)i=DATA.steps.indexOf(+cg.step);if(i<0)return;const L=i/(NS_-1)*100;h+=`<i style="left:${L}%;height:7px;background:${cg.dir>0?'#5fd0ff':'#ff6a6a'}"></i>`;});}
  ticksEl.innerHTML=h;}
 const distEl=document.getElementById('dist');let distOn=false;
 const trajEl=document.getElementById('traj');let trajOn=false;
 function closeOverlays(keep){if(keep!=='dist'&&distOn){distOn=false;distEl.classList.remove('on');document.getElementById('distbtn').classList.remove('on');}if(keep!=='traj'&&trajOn){trajOn=false;trajEl.classList.remove('on');document.getElementById('trajbtn').classList.remove('on');}}
 document.getElementById('distbtn').addEventListener('click',function(){distOn=!distOn;if(distOn)closeOverlays('dist');this.classList.toggle('on',distOn);distEl.classList.toggle('on',distOn);if(distOn)renderDist(cur);});
 function renderDist(i){if(!distOn)return;const vals=[-2,-1,0,1,2],cnt=[0,0,0,0,0];A.forEach(a=>{const v=Math.max(-2,Math.min(2,Math.round(opAt(a,i))));cnt[v+2]++;});const mx=Math.max(1,...cnt);const cols=['#d92b2b','#f59331','#8f96ad','#39c0e0','#2f6bff'];
  let h='<div style="font-size:11px;color:#8b96ad;margin-bottom:8px">distribution · step '+DATA.steps[i]+'</div><div class="db">';
  vals.forEach((v,k)=>{const ht=Math.round(cnt[k]/mx*84);h+=`<div class="col"><span class="cnt">${cnt[k]||''}</span><b style="height:${ht}px;background:${cols[k]}"></b><span>${v>0?'+':''}${v}</span></div>`;});
  distEl.innerHTML=h+'</div>';}
 const TW=296,TH=150;
 function buildTraj(){let lines='';A.forEach(a=>{let p='';a.series.forEach((v,i)=>{const x=i/(NS_-1)*TW,y=TH-((v+2)/4)*TH;p+=(i?' L':'M')+x.toFixed(1)+' '+y.toFixed(1);});lines+=`<path d="${p}" fill="none" stroke="${col(a.op)}" stroke-width="1" opacity="0.45"/>`;});
  let mp='';DATA.bdp.forEach((m,i)=>{const x=i/(NS_-1)*TW,y=TH-((+m.B+2)/4)*TH;mp+=(i?' L':'M')+x.toFixed(1)+' '+y.toFixed(1);});
  trajEl.innerHTML=`<div style="font-size:11px;color:#8b96ad;margin-bottom:6px">all agents · opinion over time</div><svg width="${TW}" height="${TH}"><line x1="0" y1="${TH/2}" x2="${TW}" y2="${TH/2}" stroke="#1b2440"/>${lines}<path d="${mp}" fill="none" stroke="#fff" stroke-width="2"/><line id="trajmk" x1="0" y1="0" x2="0" y2="${TH}" stroke="#ffe08a" stroke-width="1.5" opacity=".85"/></svg><div style="font-size:10px;color:#8b96ad;margin-top:4px">white = mean B · yellow = current step</div>`;}
 document.getElementById('trajbtn').addEventListener('click',function(){trajOn=!trajOn;if(trajOn)closeOverlays('traj');this.classList.toggle('on',trajOn);trajEl.classList.toggle('on',trajOn);if(trajOn){buildTraj();updateTrajMk(cur);}});
 function updateTrajMk(i){if(!trajOn)return;const mk=document.getElementById('trajmk');if(mk){const x=i/(NS_-1)*TW;mk.setAttribute('x1',x);mk.setAttribute('x2',x);}}
 function sec(title,inner,collapsed){return '<div class="sec sechead'+(collapsed?' collapsed':'')+'"><span class="cav">&#9660;</span>'+title+'</div><div class="secbody">'+inner+'</div>';}
 window.open_=1;function open(a){panel.classList.add('open');
  let h=`<span class="close" onclick="closePanel()">×</span>`;
  h+=`<h2><span class="dot" style="background:${col(opAt(a,cur))}"></span>${a.hub?'★ ':''}${a.name}</h2>`;
  h+=`<div class="sub">${a.hub?'hub':'agent'} · opinion now ${opAt(a,cur)} · final ${a.op}</div>`;
  h+=sec('opinion trajectory',spark(a),false);
  let chg='';if(a.changes.length)a.changes.slice().reverse().forEach(c=>{chg+=`<div class="chg"><span class="stp jump" data-step="${c.step}">step ${c.step} &#8615;</span><span class="lab">${c.from} → ${c.to}</span><span class="${c.dir>0?'up':'down'}">${c.dir>0?'▲':'▼'}</span></div>`;});else chg='<div class="sub">stable</div>';
  h+=sec(`opinion changes (${a.changes.length})`,chg,false);
  h+=sec(`tweets written (${a.written.length})`,chipsW(a)+`<div class="msglist">${rows(a.written,'w',a)}</div>`,true);
  h+=sec(`tweets read (${a.read.length})`,chipsR(a)+`<div class="msglist">${rows(a.read,'r',a)}</div>`,true);
  const drift=(a.op-a.series[0]);const fc=a.changes.length?('step '+a.changes[0].step):'never';const nbn=(a.neighbors||[]).map(id=>by[id]?by[id].name:id).join(', ');
  const moreH=`<div class="more2"><div><span class="k2">Role</span><b>${a.hub?'network hub':'peripheral'}</b></div><div><span class="k2">Degree</span><b>${a.degree||0}</b></div><div><span class="k2">Neighbours</span><b>${nbn||'—'}</b></div><div><span class="k2">Initial &#8594; final</span><b>${a.series[0]} &#8594; ${a.op}</b></div><div><span class="k2">Net drift</span><b>${drift>=0?'+':''}${drift}</b></div><div><span class="k2">First change</span><b>${fc}</b></div><div><span class="k2">Posts / reads</span><b>${a.written.length} / ${a.read.length}</b></div></div>`;
  if(a.pers){
   let ph='';
   if(a.pers.traits){for(const k in a.pers.traits){ph+=`<div><span class="k2">${k.replace(/_/g,' ')}</span><b>${a.pers.traits[k]}</b></div>`;}}
   ph='<div class="perst">'+ph+'</div>';
   if(a.pers.card){ph+=`<div class="perscard">${a.pers.card}</div>`;}
   h+=sec('persona (as configured)',ph,true);
  }
  h+=sec('additional information',moreH,true);
  pin.innerHTML=h;pin.querySelectorAll('.chips').forEach(wireChips);
  pin.querySelectorAll('.sechead').forEach(hd=>hd.addEventListener('click',()=>hd.classList.toggle('collapsed')));
  pin.querySelectorAll('.msg').forEach(el=>el.querySelector('.row').addEventListener('click',()=>el.classList.toggle('open')));
  pin.querySelectorAll('.jump').forEach(el=>el.addEventListener('click',ev=>{ev.stopPropagation();const raw=el.getAttribute('data-step');let i=DATA.steps.indexOf(+raw);if(i<0)i=DATA.steps.indexOf(raw);if(i<0)i=0;inp.value=i;setStep(i);}));
  selected=a;highlight();renderTicks(a);updateDrawer();}

 document.addEventListener('keydown',e=>{if(e.target&&e.target.tagName==='INPUT')return;if(e.key==='ArrowRight'){const n=Math.min(NS_-1,cur+1);inp.value=n;setStep(n);}else if(e.key==='ArrowLeft'){const n=Math.max(0,cur-1);inp.value=n;setStep(n);}else if(e.key===' '){e.preventDefault();if(playing)stopPlay();else startPlay(false);}});
 renderTicks(null);setStep(0);updateDrawer();
})();</script></body></html>'''

def main():
    run_dir = sys.argv[1].rstrip("/")
    data = build_data(run_dir)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out = run_dir + "_viewer.html"
    open(out, "w", encoding="utf-8").write(html)
    print("agents", len(data["agents"]), "edges", len(data["edges"]), "steps", len(data["steps"]))
    print("wrote", out, os.path.getsize(out), "bytes")

if __name__ == "__main__": main()
