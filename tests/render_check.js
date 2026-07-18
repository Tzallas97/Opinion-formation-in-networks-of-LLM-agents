// Real-three.js render check: loads the inlined three.js from an HTML file,
// stubs ONLY the GPU renderer + DOM, then builds the scene. Catches three.js
// API misuse (e.g. assigning to read-only .position) that a syntax check misses.
// Usage: node tests/render_check.js <path-to-html>
const fs=require('fs'),os=require('os'),path=require('path');
const file=process.argv[2];
if(!file){console.log('usage: node render_check.js <html>');process.exit(2);}
const html=fs.readFileSync(file,'utf8');
const blocks=[...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m=>m[1]).filter(s=>s.trim());
const threeBlock=blocks.find(b=>/\bTHREE\b/.test(b)&&b.length>100000);
const sceneBlock=blocks[blocks.length-1];
if(!threeBlock){console.log('[SKIP] render_check: no inlined three.js found in '+path.basename(file));process.exit(0);}
const grad={addColorStop(){}};
const ctx=new Proxy({},{get:(t,p)=>{if(p==='createRadialGradient'||p==='createLinearGradient')return ()=>grad;if(p==='measureText')return ()=>({width:10});return ()=>{};},set:()=>true});
const canvasLike=()=>({width:300,height:150,style:{},getContext:()=>ctx,addEventListener:()=>{}});
const elem=()=>({style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},addEventListener(){},
  appendChild(){},removeChild(){},querySelector:()=>elem(),querySelectorAll:()=>[],getContext:()=>ctx,
  _h:'',get innerHTML(){return this._h;},set innerHTML(v){this._h=v;},textContent:'',width:300,height:150,max:0,value:0});
global.self=global;global.window=global;global.innerWidth=1200;global.innerHeight=800;global.devicePixelRatio=1;
global.requestAnimationFrame=()=>{};global.addEventListener=()=>{};
global.document={getElementById:()=>canvasLike(),createElement:t=>t==='canvas'?canvasLike():elem(),querySelectorAll:()=>[]};
const tmp=path.join(os.tmpdir(),'three_inlined_'+Date.now()+'.js');fs.writeFileSync(tmp,threeBlock);
const THREE=require(tmp);
THREE.WebGLRenderer=function(){return {setPixelRatio(){},setSize(){},render(){},domElement:canvasLike()};};
global.THREE=THREE;
try{ eval(sceneBlock); console.log('[PASS] render_check: '+path.basename(file)+' builds against real three.js'); process.exit(0); }
catch(e){ console.log('[FAIL] render_check: '+path.basename(file)); console.log('   '+String((e&&e.stack||e)).split('\n').slice(0,3).join('\n   ')); process.exit(1); }
finally{ try{fs.unlinkSync(tmp);}catch(_){} }
