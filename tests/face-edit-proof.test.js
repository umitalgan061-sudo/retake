import {test} from 'node:test';import assert from 'node:assert/strict';import {measureFaceEdit} from '../src/quality.js';
test('unchanged image and outer-rim-only changes cannot certify a face edit',()=>{
 const w=100,h=100,src=new Uint8ClampedArray(w*h*4).fill(120),out=src.slice(),b={x:20,y:20,width:60,height:60};
 assert.equal(measureFaceEdit(src,out,w,h,b).meaningful,false);
 for(let y=20;y<80;y++)for(let x=20;x<80;x++)if(x<24||x>75||y<24||y>75)out[(y*w+x)*4]=190;
 assert.equal(measureFaceEdit(src,out,w,h,b).meaningful,false);
 for(let y=35;y<65;y++)for(let x=35;x<65;x++)out[(y*w+x)*4]=140;
 assert.equal(measureFaceEdit(src,out,w,h,b).meaningful,true);
});
