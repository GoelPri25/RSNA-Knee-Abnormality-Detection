const fs=require('fs');
const {chromium}=require('playwright-core');
const root='./data/local_global/architecture_illustration';
(async()=>{
  const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  const page=await browser.newPage({viewport:{width:768,height:950}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('file://'+root+'/preview-local-global-reader.html');
  const frame=page.frameLocator('iframe');const el=frame.locator('#knee-local-global-reader');
  await el.waitFor();await frame.locator('#klg-prev').waitFor();
  await frame.locator('#klg-mode').selectOption('eval');
  await frame.locator('#klg-series').selectOption('2');
  await frame.locator('#klg-input').selectOption('support');
  if(!await frame.locator('#klg-local-pool').textContent().then(x=>x.includes('5 of 48')))throw Error('Mode interaction failed');
  if(!await frame.locator('#klg-prev').getAttribute('alt').then(x=>x.includes('Axial')&&x.includes('support')))throw Error('Series/input interaction failed');
  await frame.locator('#klg-input').selectOption('images');await frame.locator('#klg-series').selectOption('0');
  await frame.locator('#klg-mode').selectOption('train');
  async function screenshot(name){
    const height=await el.evaluate(r=>r.scrollHeight);
    await page.locator('iframe').evaluate((f,h)=>f.style.height=(h+32)+'px',height);
    await page.setViewportSize({width:page.viewportSize().width,height:height+64});
    await el.screenshot({path:root+'/'+name});
  }
  await screenshot('preview-768.png');
  await page.setViewportSize({width:360,height:950});
  await screenshot('preview-360.png');
  const overflow=await el.evaluate(r=>r.scrollWidth>r.clientWidth+1);
  const imageCheck=await frame.locator('.slices img').evaluateAll(imgs=>imgs.every(i=>i.complete&&i.naturalWidth===224));
  if(errors.length||overflow||!imageCheck)throw Error(JSON.stringify({errors,overflow,imageCheck}));
  fs.writeFileSync(root+'/visual_check.json',JSON.stringify({errors,overflow,imageCheck,mode_and_series_controls:true,widths:[768,360]},null,2));
  console.log('Images, controls, responsive layout and browser execution checked.');
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
