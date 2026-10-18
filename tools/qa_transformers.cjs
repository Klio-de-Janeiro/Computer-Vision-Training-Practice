// Run against an isolated local HTTP copy; never attach to a user's browser or storage.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {chromium}=require(process.argv[2]||'playwright');
const root=path.resolve(__dirname,'../dist');
const output=path.resolve(__dirname,'../qa');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff2':'font/woff2','.woff':'font/woff','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml'};
const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname.replace(/\/$/,'/index.html')));
  if(!file.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
  fs.readFile(file,(err,body)=>{res.writeHead(err?404:200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});res.end(err?'Missing file':body);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base='http://127.0.0.1:'+server.address().port;
  let browser;
  try{
    browser=await chromium.launch({headless:true,executablePath:process.argv[3]||undefined});
    const context=await browser.newContext({viewport:{width:1440,height:1100}});
    const page=await context.newPage();
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await context.route('**/*',route=>{
      const url=route.request().url();
      if(!url.startsWith(base+'/')&&!url.startsWith('data:')&&!url.startsWith('blob:')){errors.push('Unexpected request: '+url);return route.abort();}
      return route.continue();
    });
    await page.goto(base+'/index.html');
    await page.locator('.lesson-card').first().waitFor();
    const banks=await page.evaluate(()=>CV_CONTENT.architectures);
    const go=async hash=>{await page.evaluate(h=>{location.hash=h;},hash);await page.waitForTimeout(60);};
    for(const a of banks){
      await go('#theory?topic='+a.topic+'&lesson='+a.lesson);
      assert.equal(await page.locator('.architecture-theory h2').count(),a.sections.length,a.id);
      assert.equal(await page.locator('.katex-error').count(),0,a.id);
      await go('#practice?architecture='+a.id+'&count=all');
      assert.equal(await page.locator('#architecture-filter').inputValue(),a.id);
      assert.equal(await page.locator('#lesson-filter').inputValue(),String(a.lesson));
      await page.locator('#practice-form button[type=submit]').click();
      if(await page.locator('#replace-session').isVisible())await page.locator('#confirm-start').click();
      await page.locator('#answer-form').waitFor();
      const session=await page.evaluate(()=>({ids:state.session.ids,questions:state.session.ids.map(id=>CV_CONTENT.questions.find(q=>q.id===id))}));
      assert.ok(session.ids.length>=30,a.id);
      assert.ok(session.questions.every(q=>q.architecture===a.id),a.id);
    }
    await go('#practice?topic=rope&lesson=4');
    await page.locator('#architecture-filter').selectOption('bert');
    assert.equal(await page.locator('input[name=topic]').inputValue(),'');
    assert.match(await page.locator('#pool-size').textContent(),/32/);
    assert.equal(await page.locator('#topic-restriction').isVisible(),false);
    await page.locator('#lesson-filter').selectOption('1');
    assert.equal(await page.locator('#architecture-filter').inputValue(),'');
    assert.match(await page.locator('#pool-size').textContent(),/24/);
    await go('#practice?architecture=swin&type=open&count=5');
    await page.locator('#practice-form button[type=submit]').click();
    await page.locator('#confirm-start').click();
    await page.locator('#answer-text').fill('Сдвиг меняет границы окон и связывает соседние группы.');
    await page.locator('#answer-form button[type=submit]').click();
    await page.locator('.feedback').waitFor();
    assert.equal(await page.locator('.rubric li').count(),3);
    await page.locator('[data-rate="0.5"]').click();
    const saved=await page.evaluate(()=>JSON.parse(JSON.stringify(state)));
    await page.locator('#theme-select').selectOption('neon-light');
    await page.reload();await page.locator('.feedback').waitFor();
    assert.deepEqual(await page.evaluate(()=>JSON.parse(JSON.stringify(state))),saved);
    assert.equal(await page.locator('html').getAttribute('data-theme'),'neon-light');
    fs.mkdirSync(output,{recursive:true});
    for(const theme of ['neon-light','neon-dark']){
      await page.locator('#theme-select').selectOption(theme);
      await go('#lesson?lesson=4');
      assert.equal(await page.locator('.architecture-card').count(),10);
      await page.screenshot({path:path.join(output,'transformers-'+theme+'.png'),fullPage:true});
      await page.setViewportSize({width:390,height:844});
      await page.waitForTimeout(300);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      await page.screenshot({path:path.join(output,'transformers-mobile-'+theme+'.png'),fullPage:true});
      await page.screenshot({path:path.join(output,'transformers-mobile-viewport-'+theme+'.png')});
      await go('#practice?architecture=swin&count=all');
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      await page.setViewportSize({width:1440,height:1100});
    }
    await go('#theory?topic=swin&lesson=4');
    await page.screenshot({path:path.join(output,'swin-expanded-theory.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('Transformer UI passed: 11 banks, full sessions, expanded theory, filter transitions, answers, persistence, both neon themes, mobile.');
  }finally{
    if(browser)await browser.close();
    await new Promise(resolve=>server.close(resolve));
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
