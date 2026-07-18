#!/usr/bin/env python3
"""Side-by-side A/B viewer: two runs, one synchronized step scrubber.

Usage: python tools/build_ab_viewer.py RUN_A RUN_B [--labels a,b] [--out FILE]
Writes an offline HTML with the two networks left/right, shared play/scrub,
per-side B/D/P + distribution, and click-to-inspect agent sparklines.
For full per-agent tweet drill-down use the single-run viewer (build_viewer.py).
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_viewer import build_data  # noqa: E402

TEMPLATE = r'''<!DOCTYPE html><html><head><meta charset="utf-8"/><title>A/B viewer</title><style>
html,body{margin:0;height:100%;background:#06080f;color:#c8d2e6;font-family:-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
.cols{display:flex;height:calc(100vh - 64px)}
.pane{flex:1;position:relative;border-right:1px solid #16203a}.pane:last-child{border-right:0}
svg{width:100%;height:100%}
.pt{position:absolute;top:10px;left:14px;font-size:13px;color:#eaf1ff;font-weight:600}
.hud{position:absolute;top:34px;left:14px;background:rgba(10,14,26,.75);border:1px solid #1e2740;border-radius:8px;padding:7px 10px;font-size:12px;display:flex;gap:14px}
.hud b{display:block;font-size:14px;color:#eaf1ff}
.dist{position:absolute;bottom:12px;left:14px;background:rgba(10,14,26,.85);border:1px solid #1e2740;border-radius:8px;padding:8px;display:flex;gap:5px;align-items:flex-end;height:64px}
.dist .col{width:20px;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.dist .col b{width:100%;border-radius:2px 2px 0 0;min-height:2px}.dist .col span{font-size:9px;color:#8b96ad;margin-top:2px}
.bar{display:flex;align-items:center;gap:12px;height:64px;background:#0a0f1c;border-top:1px solid #1e2740;padding:0 16px}
.pbtn{background:#1e2b4d;border:0;color:#fff;width:30px;height:30px;border-radius:50%;cursor:pointer}
.lbl{font-size:12.5px;color:#b9c4dc;min-width:130px}
input[type=range]{flex:1}
.info{position:absolute;right:12px;top:34px;width:250px;background:rgba(9,13,24,.96);border:1px solid #1e2740;border-radius:9px;padding:10px 12px;font-size:12px;display:none}
.info h3{margin:0 0 4px;font-size:13px;color:#eaf1ff}.info .sub{color:#7d88a3;font-size:11px;margin-bottom:6px}
.tip{position:absolute;pointer-events:none;background:rgba(6,10,20,.96);border:1px solid #2d3f6b;border-radius:6px;padding:3px 7px;font-size:11px;color:#eaf1ff;display:none;z-index:5}
.note{position:absolute;bottom:84px;right:14px;font-size:10.5px;color:#5b678a}
</style></head><body>
<div class="cols" id="cols"></div>
<div class="bar"><button class="pbtn" id="play">&#9654;</button><span class="lbl" id="slbl"></span><input type="range" id="step" min="0" value="0"/></div>
<div class="tip" id="tip"></div>
<script>const RUNS=__DATA__;const LABELS=__LABELS__;
(function(){
 const NSs=RUNS.map(r=>r.steps.length), NS=Math.max(...NSs);
 const cols=document.getElementById('cols');
 const OPCOL={'-2':'#d92b2b','-1':'#f59331','0':'#8f96ad','1':'#39c0e0','2':'#2f6bff'};
 const col=op=>OPCOL[Math.max(-2,Math.min(2,Math.round(op)))]||'#8f96ad';
 const panes=[];
 RUNS.forEach((R,pi)=>{
  const pane=document.createElement('div');pane.className='pane';
  pane.innerHTML=`<div class="pt">${LABELS[pi]}</div><div class="hud" id="hud${pi}"></div>`+
   `<svg id="svg${pi}" viewBox="0 0 800 560" preserveAspectRatio="xMidYMid meet"></svg>`+
   `<div class="dist" id="dist${pi}"></div><div class="info" id="info${pi}"></div>`+
   `<div class="note">click a node for its trajectory</div>`;
  cols.appendChild(pane);
  const svg=pane.querySelector('svg'),NSvg='http://www.w3.org/2000/svg';
  const A=R.agents,by={};A.forEach(a=>by[a.id]=a);
  const xs=A.map(a=>a.x),ys=A.map(a=>a.y),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  A.forEach(a=>{a.px=60+((a.x-x0)/((x1-x0)||1))*680;a.py=50+((a.y-y0)/((y1-y0)||1))*460;});
  R.edges.forEach(e=>{const a=by[e[0]],b=by[e[1]];if(!a||!b)return;const l=document.createElementNS(NSvg,'line');
   l.setAttribute('x1',a.px);l.setAttribute('y1',a.py);l.setAttribute('x2',b.px);l.setAttribute('y2',b.py);
   l.setAttribute('stroke','#26324f');l.setAttribute('stroke-width','1.1');svg.appendChild(l);});
  const arrow=document.createElementNS(NSvg,'line');arrow.setAttribute('stroke','#ffe08a');arrow.setAttribute('stroke-width','2.2');arrow.setAttribute('opacity','0');svg.appendChild(arrow);
  const nodeEls={};
  A.forEach(a=>{const g=document.createElementNS(NSvg,'g');g.style.cursor='pointer';
   const c=document.createElementNS(NSvg,'circle');c.setAttribute('cx',a.px);c.setAttribute('cy',a.py);c.setAttribute('r',a.hub?13:8);c.setAttribute('stroke','#0a0f1c');c.setAttribute('stroke-width','2');g.appendChild(c);
   const t=document.createElementNS(NSvg,'text');t.setAttribute('x',a.px);t.setAttribute('y',a.py-(a.hub?18:12));t.setAttribute('text-anchor','middle');t.setAttribute('fill','#aeb9d2');t.setAttribute('font-size','9');t.textContent=(a.hub?'★ ':'')+a.name;g.appendChild(t);
   g.addEventListener('click',()=>showInfo(pi,a));
   g.addEventListener('mousemove',ev=>{const tp=document.getElementById('tip');tp.style.display='block';tp.style.left=(ev.clientX+10)+'px';tp.style.top=(ev.clientY+10)+'px';tp.textContent=a.name+' · '+opAt(R,a,cur);});
   g.addEventListener('mouseleave',()=>{document.getElementById('tip').style.display='none';});
   svg.appendChild(g);nodeEls[a.id]=c;});
  panes.push({R,A,nodeEls,arrow,pi});
 });
 function opAt(R,a,i){const k=Math.min(i,a.series.length-1);return a.series[k];}
 function bdpAt(R,i){return R.bdp[Math.min(i,R.bdp.length-1)];}
 function renderPane(p,i){
  const {R,A,nodeEls,arrow,pi}=p;
  A.forEach(a=>{const c=nodeEls[a.id];c.setAttribute('fill',col(opAt(R,a,i)));
   let d=0;if(i>0){const cu=opAt(R,a,i),pr=opAt(R,a,i-1);d=cu>pr?1:cu<pr?-1:0;}
   c.setAttribute('stroke',d>0?'#5fd0ff':d<0?'#ff6a6a':'#0a0f1c');c.setAttribute('stroke-width',d?'3':'2');});
  const m=bdpAt(R,i);
  document.getElementById('hud'+pi).innerHTML=`<div>B<b>${(+m.B).toFixed(2)}</b></div><div>D<b>${(+m.D).toFixed(2)}</b></div><div>P<b>${(+m.P).toFixed(2)}</b></div><div>step<b>${R.steps[Math.min(i,R.steps.length-1)]}</b></div>`;
  const st=R.steps[Math.min(i,R.steps.length-1)];const it=R.interactions.find(x=>x.step==st);
  if(it){const fa=A.find(a=>a.name===it.from),la=A.find(a=>a.name===it.to);
   if(fa&&la){arrow.setAttribute('x1',fa.px);arrow.setAttribute('y1',fa.py);arrow.setAttribute('x2',la.px);arrow.setAttribute('y2',la.py);arrow.setAttribute('opacity','.8');}
  } else arrow.setAttribute('opacity','0');
  const cnt=[0,0,0,0,0];A.forEach(a=>{cnt[Math.max(-2,Math.min(2,Math.round(opAt(R,a,i))))+2]++;});
  const mx=Math.max(1,...cnt),cols5=['#d92b2b','#f59331','#8f96ad','#39c0e0','#2f6bff'];
  document.getElementById('dist'+pi).innerHTML=[-2,-1,0,1,2].map((v,k)=>`<div class="col"><b style="height:${Math.round(cnt[k]/mx*40)}px;background:${cols5[k]}"></b><span>${v>0?'+':''}${v}</span></div>`).join('');
 }
 function showInfo(pi,a){
  const R=RUNS[pi],el=document.getElementById('info'+pi);
  const W=220,H=40,n=a.series.length;let path='';
  a.series.forEach((v,i)=>{const x=i/(n-1)*W,y=H-((v+2)/4)*H;path+=(i?' L':'M')+x.toFixed(1)+' '+y.toFixed(1);});
  let dots='';(a.changes||[]).forEach(c=>{let i=R.steps.indexOf(c.step);if(i<0)i=R.steps.indexOf(+c.step);if(i<0)return;const x=i/(n-1)*W,y=H-((c.to+2)/4)*H;dots+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.8" fill="${c.dir>0?'#5fd0ff':'#ff6a6a'}"/>`;});
  el.innerHTML=`<h3>${a.hub?'★ ':''}${a.name}</h3><div class="sub">${a.series[0]} → ${a.op} · ${(a.changes||[]).length} changes</div>`+
   `<svg width="${W}" height="${H}"><line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}" stroke="#1b2440"/><path d="${path}" fill="none" stroke="#c8d2e6" stroke-width="1.4"/>${dots}</svg>`+
   `<div class="sub" style="margin-top:6px">full detail: open this run's single viewer</div>`;
  el.style.display='block';
 }
 let cur=0,playing=false,loop=false,timer=null,pressT=null,longP=false;
 const inp=document.getElementById('step');inp.max=NS-1;
 const pb=document.getElementById('play');
 function setStep(i){cur=i;document.getElementById('slbl').textContent='Step '+(i)+' / '+(NS-1);panes.forEach(p=>renderPane(p,i));}
 inp.addEventListener('input',e=>setStep(+e.target.value));
 function stopPlay(){playing=false;clearInterval(timer);pb.innerHTML='&#9654;';}
 function startPlay(lp){loop=lp;playing=true;pb.innerHTML=lp?'&#9632;':'&#10074;&#10074;';clearInterval(timer);
  timer=setInterval(()=>{let n=cur+1;if(n>=NS){if(loop){n=0;}else{stopPlay();return;}}inp.value=n;setStep(n);},350);}
 pb.title='Click: play once · Hold: loop';
 pb.addEventListener('mousedown',()=>{longP=false;pressT=setTimeout(()=>{longP=true;startPlay(true);},400);});
 pb.addEventListener('mouseup',()=>{clearTimeout(pressT);if(longP)return;if(playing)stopPlay();else startPlay(false);});
 pb.addEventListener('mouseleave',()=>clearTimeout(pressT));
 document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){const n=Math.min(NS-1,cur+1);inp.value=n;setStep(n);}else if(e.key==='ArrowLeft'){const n=Math.max(0,cur-1);inp.value=n;setStep(n);}else if(e.key===' '){e.preventDefault();if(playing)stopPlay();else startPlay(false);}});
 setStep(0);
})();</script></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b")
    ap.add_argument("--labels", default=None, help="comma-separated pane titles")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [build_data(a.run_a.rstrip("/\\")), build_data(a.run_b.rstrip("/\\"))]
    labels = [x.strip() for x in a.labels.split(",")] if a.labels else [r["run"] for r in runs]
    out = a.out or os.path.join(os.path.dirname(a.run_a.rstrip("/\\")) or ".",
                                f"ab_viewer_{labels[0]}_VS_{labels[1]}.html".replace(" ", "_"))
    html = TEMPLATE.replace("__DATA__", json.dumps(runs, ensure_ascii=False)).replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
