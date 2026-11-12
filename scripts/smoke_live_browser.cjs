// Real H.264 -> Chromium video callbacks -> cropped canvas + delayed metadata.
const {spawn}=require('node:child_process'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),assert=require('node:assert/strict');
const browser=process.env.CHROME_BIN||'C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'edge-timing-browser-'));
const fixture=spawn(process.env.PYTHON||'python',['-m','scripts.browser_timing_fixture'],{cwd:path.join(__dirname,'..'),stdio:['ignore','pipe','pipe']});
fixture.stderr.on('data',data=>process.stderr.write(data));
const chrome=spawn(browser,['--headless=new','--remote-debugging-port=0',`--user-data-dir=${profile}`,'--autoplay-policy=no-user-gesture-required','--no-first-run','--disable-background-timer-throttling','--disable-renderer-backgrounding','about:blank'],{stdio:['ignore','pipe','pipe']});
let socket;
function output(child,stream,pattern){return new Promise((resolve,reject)=>{
  let text='';const timeout=setTimeout(()=>reject(Error('startup timed out: '+text.slice(-1000))),20000);
  child.on('error',reject);child.on('exit',code=>reject(Error('process exited '+code+': '+text.slice(-1000))));
  child[stream].on('data',data=>{text+=data;const match=text.match(pattern);if(match){clearTimeout(timeout);resolve(match[1]);}});
});}
(async()=>{
  const [port,endpoint]=await Promise.all([output(fixture,'stdout',/^(\d+)/),output(chrome,'stderr',/DevTools listening on (ws:\/\/\S+)/)]);
  console.log('Fixture and headless browser started');
  socket=new WebSocket(endpoint);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let id=0;const pending=new Map(),errors=[];
  socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result);}
    if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text+JSON.stringify(m.params.exceptionDetails.exception));};
  function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const n=++id;const timer=setTimeout(()=>reject(Error('CDP timeout: '+method)),10000);pending.set(n,{resolve:value=>{clearTimeout(timer);resolve(value);},reject:e=>{clearTimeout(timer);reject(e);}});socket.send(JSON.stringify({id:n,method,params,sessionId}));});}
  const {targetId}=await call('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
  await call('Page.enable',{},sessionId);await call('Runtime.enable',{},sessionId);
  await call('Page.navigate',{url:`http://127.0.0.1:${port}/`},sessionId);
  await new Promise(resolve=>setTimeout(resolve,5000));
  const result=await call('Runtime.evaluate',{expression:'JSON.stringify({counts:timeline.counts,invalidFrames,seq:timeline.lastShown,width:display.width,height:display.height,mediaTime:video.currentTime,status:label("match").textContent})',returnByValue:true},sessionId);
  const report=JSON.parse(result.result.value);
  assert.deepEqual(errors,[]);assert.equal(report.invalidFrames,0);assert.equal(report.width,320);assert.equal(report.height,180);
  assert.ok(report.counts.exact>=30,JSON.stringify(report));assert.equal(report.counts.unmatched,0);assert.ok(report.seq>=740);
  assert.ok(report.mediaTime<5);assert.ok(report.status.includes('overlay expired'));
  console.log(JSON.stringify({test_status:'PASS',...report}));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>{
  socket?.close();chrome.kill();fixture.kill();
  // Isolated browser profile deliberately retained for inspection; never touch
  // the user's existing Chrome profile.
  console.log('Temporary browser profile: '+profile);
});
