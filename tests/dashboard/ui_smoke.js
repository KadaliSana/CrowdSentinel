/**
 * Runtime smoke test for the dashboard front-end.  Run: node tests/dashboard/ui_smoke.js
 * (from the repo root; exits non-zero on failure).
 *
 * `node --check` only proves main.js PARSES. This executes it against a
 * stubbed DOM and drives applySnapshot() through the states that matter,
 * asserting the one contract the whole UI rests on: unknown is never
 * rendered as zero, as calm, or as a low percentage. A dashboard that shows
 * "0 people, 0.0000 s^-2, NORMAL" when it has actually stopped being able to
 * see is the specific failure this project exists to prevent.
 *
 * The stub is deliberately thin. Where it diverges from a real DOM the
 * assertions work around it explicitly (a DocumentFragment append looks like
 * one child; textContent is not coerced to a string).
 */

// Minimal DOM stub: enough to execute main.js top-level and drive applySnapshot.
const ids = require('fs').readFileSync(__dirname + '/../../src/dashboard/templates/index.html','utf8')
  .match(/id="[^"]+"/g).map(s => s.slice(4,-1));
const made = {};
function el(id){
  if (made[id]) return made[id];
  const e = {
    id, hidden:false, textContent:'', innerHTML:'', value:'',
    style:(()=>{const store={};return new Proxy(store,{get:(t,k)=>k==='setProperty'?((n,v)=>{t[n]=v}):k==='removeProperty'?((n)=>{delete t[n]}):k==='getPropertyValue'?((n)=>t[n]||''):(t[k]||''),set:(t,k,v)=>{t[k]=v;return true}});})(),
    dataset:{}, children:[], _cls:new Set(),
    classList:{add(c){e._cls.add(c)},remove(c){e._cls.delete(c)},
               toggle(c,on){on?e._cls.add(c):e._cls.delete(c)},
               contains(c){return e._cls.has(c)}},
    setAttribute(){}, removeAttribute(){}, getAttribute(){return null},
    appendChild(c){e.children.push(c);return c}, replaceChildren(){e.children=[]},
    addEventListener(){}, removeEventListener(){}, focus(){},
    getBoundingClientRect(){return {width:640,height:360,top:0,left:0}},
    getContext(){return {canvas:{}}},
    querySelectorAll(){return []}, querySelector(){return null},
    remove(){}, insertAdjacentHTML(){},
    naturalWidth:1280, naturalHeight:704, clientWidth:640, clientHeight:352,
  };
  return made[id]=e;
}
ids.forEach(el);
global.document = {
  getElementById:(id)=> made[id] || null,
  createElement:()=> el('_tmp'+Math.random()),
  createDocumentFragment:()=> el('_frag'+Math.random()),
  addEventListener(){}, querySelectorAll(){return []}, querySelector(){return null},
  documentElement: el('_root'), body: el('_body'),
};
global.window = { matchMedia:()=>({matches:false, addEventListener(){}}),
                  addEventListener(){}, requestAnimationFrame:(f)=>f() };
global.EventSource = class { constructor(){ this.readyState=0; } close(){} };
global.Chart = class { constructor(ctx, config){ config = config || {}; this.config = config; this.data = config.data || {labels:[],datasets:[]}; this.options = config.options || {}; } update(){} destroy(){} resize(){} };
global.fetch = async () => ({ ok:true, json: async()=>({}) });
global.requestAnimationFrame = (f)=>f();
global.setInterval = ()=>0; global.setTimeout = ()=>0;

const src = require('fs').readFileSync('/home/sana/Yan/src/dashboard/static/main.js','utf8');
require('vm').runInThisContext(src);
console.log('TOP-LEVEL OK');

const base = {status:'live',status_text:'LIVE',detail:'',known:true,count:3,
  count_reason:null,frames:100,dropped:5,channel:'camstream',region:'ap-south-1',
  pressure:0.0012,max_pressure:0.031,level:'high',coverage:1.0,
  last_alert:{level:'high',previous_level:'normal',reason:'escalation',
              message:'normal -> high',timestamp:12.5},
  thresholds:{elevated:0.010,high:0.020,critical:0.040},
  grid_shape:[11,20],
  cells:Array.from({length:11},(_,r)=>Array.from({length:20},(_,c)=> (r+c)%7===0?null:0.001*(r+c)))};

const cases = {
  'live/high': base,
  'critical': {...base, level:'critical', max_pressure:0.09},
  'unknown+null count': {...base, level:'unknown', known:false, count:null,
      count_reason:'board sent no count', pressure:null, max_pressure:null,
      coverage:null, cells:null, grid_shape:null},
  'idle standby': {...base, status:'idle', status_text:'SYSTEM STANDBY',
      known:false, count:null, pressure:null, max_pressure:null, cells:null,
      grid_shape:null, last_alert:null, level:'normal'},
  'no_video': {...base, status:'no_video', cells:null, grid_shape:null},
};
for (const [name, snap] of Object.entries(cases)) {
  try { applySnapshot(snap); console.log(`  ${name.padEnd(20)} OK`); }
  catch (e) { console.log(`  ${name.padEnd(20)} FAIL: ${e.message}`); process.exitCode = 1; }
}
console.log('count text:', made.personCount.textContent,
            '| pressure:', made.pressureVal.textContent,
            '| level:', made.levelWord.textContent);

// ── the contract that matters: unknown must never look like zero/calm ──
let fails = 0;
function check(label, cond, got) {
  if (cond) { console.log(`  PASS  ${label}`); }
  else { console.log(`  FAIL  ${label} (got ${JSON.stringify(got)})`); fails++; }
}

applySnapshot(cases['unknown+null count']);
check('unknown count is a dash, not 0',
      made.personCount.textContent === '—', made.personCount.textContent);
check('unknown pressure is a dash, not 0.0000',
      made.pressureVal.textContent === '—', made.pressureVal.textContent);
check('unknown coverage is a dash',
      made.coverageVal.textContent === '—', made.coverageVal.textContent);
check('level word says unknown',
      /unknown/i.test(made.levelWord.textContent), made.levelWord.textContent);
check('unknown pressure is flagged for styling',
      made.pressureVal.classList.contains('is-unknown') &&
      made.scaleTrack.classList.contains('is-unknown'),
      [[...made.pressureVal._cls], [...made.scaleTrack._cls]]);
check('the needle is hidden when there is nothing to point at',
      made.scaleNeedle.classList.contains('is-hidden'), [...made.scaleNeedle._cls]);

applySnapshot(cases['live/high']);
check('a real 0 count still renders as 0 (board saw nobody)',
      (applySnapshot({...cases['live/high'], count:0}),
       String(made.personCount.textContent) === '0'),
      made.personCount.textContent);

applySnapshot(cases['live/high']);
// cells go in via a DocumentFragment, so the stub sees one append; the
// grid note is the observable proof the lattice was built to shape.
check('lattice built to grid_shape',
      made.gridNote.textContent === 'grid 20 × 11', made.gridNote.textContent);
const cls = made.lattice.children.flatMap(r => (r.children||[]).map(c => [...c._cls]));
const flat = cls.length ? cls : made.lattice.children.map(c => [...c._cls]);
check('null cells are marked unmeasured, not coloured',
      flat.some(c => c.includes('cell--unmeasured')), flat.slice(0,3));
check('a peak cell is marked',
      flat.some(c => c.includes('cell--peak')), 'none');

applySnapshot(cases['critical']);
check('critical raises the klaxon', made.criticalOverlay.hidden === false,
      made.criticalOverlay.hidden);

console.log(fails ? `\n${fails} CONTRACT FAILURE(S)` : '\nall contracts hold');
process.exitCode = fails ? 1 : 0;
