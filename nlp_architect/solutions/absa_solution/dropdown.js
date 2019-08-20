/* *****************************************************************************
Copyright 2017-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
****************************************************************************** */

/* this file is based on example from the bokeh library: https://github.com/bokeh/bokeh/blob/master/examples/app/export_csv/download.js
 with slight changes (content being downloaded is changed).

 Bokeh license: https://github.com/bokeh/bokeh/blob/master/LICENSE.txt

 Copyright (c) 2012, Anaconda, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of Anaconda nor the names of any contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

*/


let clk = clicked.value;
let tab = tabs.tabs[tabs.active].title;

if (['open'].indexOf(clicked.value) >= 0) {
    text_status.visible = false;
    let input = document.getElementById('inputOS')
    function read_file(filename) {
        let reader = new FileReader();
        reader.onload = load_handler;
        reader.onerror = error_handler;
        // readAsDataURL represents the file's data as a base64 encoded string
        reader.readAsDataURL(filename);
    }

    function load_handler(event) {
        let b64string = event.target.result;
        let file = {'file_contents' : [b64string], 'file_name': [input.files[0].name]};

        // To enable input.onchange event fire even with the same file.
        input.value = ''

        switch(clk) {
            case 'open':
                if (tab === 'Aspect Lexicon'){
                    asp_src.data = file;
                    asp_src.trigger("change");
                }
                if (tab === 'Opinion Lexicon'){
                    op_src.data = file;
                    op_src.trigger("change");
                }
            break;
            default:
        }
    }

    function error_handler(evt) {
        if(evt.target.error.name === "NotReadableError") {
          alert("Can't read file!");
        }
    }

   /* let input = document.createElement('input');
    input.setAttribute('type', 'file');
    input.value = ""
    input.addEventListener("change", handleFiles2, false);*/
    
    function handleFiles2(){
        
        console.log("inside Input change event")
        if (window.FileReader) {
            if (input.files[0] != undefined){
                read_file(input.files[0]);
            }
        }
    }
    input.onchange = function(){
        handleFiles2();
    }
    input.removeAttribute("webkitdirectory")
    input.removeAttribute("multiple")
    /*input.onchange = function(){
        if (window.FileReader) {
            read_file(input.files[0]);
        }
    };

    input.onclick = function(){
        input.value = ""
    };*/

    input.click();
    //handleFiles2();
    clicked.value = ""
}

if (clicked.value === "save"){
    let data;
    let filetext;
    text_status.visible = false;
    if (tab === 'Aspect Lexicon'){
        data = asp_filter.data;
        filetext = 'Term,Alias1,Alias2,Alias3,Example1,Example2,Example3,Example4,Example5,Example6,Example7,Example8,Example9,Example10,Example11,Example12,Example13,Example14,Example15,Example16,Example17,Example18,Example19,Example20\n';
    }
    if (tab === 'Opinion Lexicon'){
        data = op_filter.data;
        filetext = 'Term,Score,Polarity,isAcquired,Example1,Example2,Example3,Example4,Example5,Example6,Example7,Example8,Example9,Example10,Example11,Example12,Example13,Example14,Example15,Example16,Example17,Example18,Example19,Example20\n';
    }
    if(data['Example1'] != undefined)
    {
        for (let i = 0; i < data['Term'].length; i++) {
            let currRow = [data['Term'][i],
                (tab === 'Aspect Lexicon' ? data['Alias1'][i] : data['Score'][i]) ,
                (tab === 'Aspect Lexicon' ? data['Alias2'][i] : data['Polarity'][i]) ,
                (tab === 'Aspect Lexicon' ? data['Alias3'][i] : data['isAcquired'][i]) ,
                '"' + data['Example1'][i] + '"',
                '"' + data['Example2'][i] + '"',
                '"' + data['Example3'][i] + '"',
                '"' + data['Example4'][i] + '"',
                '"' + data['Example5'][i] + '"',
                '"' + data['Example6'][i] + '"',
                '"' + data['Example7'][i] + '"',
                '"' + data['Example8'][i] + '"',
                '"' + data['Example9'][i] + '"',
                '"' + data['Example10'][i] + '"',
                '"' + data['Example11'][i] + '"',
                '"' + data['Example12'][i] + '"',
                '"' + data['Example13'][i] + '"',
                '"' + data['Example14'][i] + '"',
                '"' + data['Example15'][i] + '"',
                '"' + data['Example16'][i] + '"',
                '"' + data['Example17'][i] + '"',
                '"' + data['Example18'][i] + '"',
                '"' + data['Example19'][i] + '"',
                '"' + data['Example20'][i] + '"'].filter(x => x!='"NaN"').concat('\n');

            let joined = currRow.join();
            joined = (tab === 'Aspect Lexicon' ? joined.replace(/"AS"/g, '""AS""') :
                joined.replace(/"OP"/g, '""OP""'));

            filetext = filetext.concat(joined);
        }
        if (tab === 'Opinion Lexicon'){
            for (let i = 0; i < opinion_lex_generic['Term'].length; i++){
                let currRow = [opinion_lex_generic['Term'][i],
                    opinion_lex_generic['Score'][i] ,
                    opinion_lex_generic['Polarity'][i] ,
                    opinion_lex_generic['isAcquired'][i] ,
                    '"' + opinion_lex_generic['Example1'][i] + '"',
                    '"' + opinion_lex_generic['Example2'][i] + '"',
                    '"' + opinion_lex_generic['Example3'][i] + '"',
                    '"' + opinion_lex_generic['Example4'][i] + '"',
                    '"' + opinion_lex_generic['Example5'][i] + '"',
                    '"' + opinion_lex_generic['Example6'][i] + '"',
                    '"' + opinion_lex_generic['Example7'][i] + '"',
                    '"' + opinion_lex_generic['Example8'][i] + '"',
                    '"' + opinion_lex_generic['Example9'][i] + '"',
                    '"' + opinion_lex_generic['Example10'][i] + '"',
                    '"' + opinion_lex_generic['Example11'][i] + '"',
                    '"' + opinion_lex_generic['Example12'][i] + '"',
                    '"' + opinion_lex_generic['Example13'][i] + '"',
                    '"' + opinion_lex_generic['Example14'][i] + '"',
                    '"' + opinion_lex_generic['Example15'][i] + '"',
                    '"' + opinion_lex_generic['Example16'][i] + '"',
                    '"' + opinion_lex_generic['Example17'][i] + '"',
                    '"' + opinion_lex_generic['Example18'][i] + '"',
                    '"' + opinion_lex_generic['Example19'][i] + '"',
                    '"' + opinion_lex_generic['Example20'][i] + '"'].filter(x => x!='"NaN"').concat('\n');
    
                let joined = currRow.join();
                joined = joined.replace(/"OP"/g, '""OP""');
                filetext = filetext.concat(joined);
            }
        }
        let filename = (tab === 'Aspect Lexicon' ? 'aspect_lexicon.csv' : 'opinion_lexicon.csv');
        let blob = new Blob([filetext], {type: 'text/csv;charset="ISO-8859-1";'});

        //addresses IE
        if (navigator.msSaveBlob) {
            navigator.msSaveBlob(blob, filename);
        } else {
            var link = document.createElement("a");
            link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.target = "_blank";
            link.style.visibility = 'hidden';
            link.dispatchEvent(new MouseEvent('click'))
        }
    }
    clicked.value = ""
}

if (train_clicked.value == "parsed" || train_clicked.value == "raw"){
    functionType = "train"
    dataType = train_clicked.value
    text_status.value = 'Select data for lexicon extraction';
}
if (infer_clicked.value == "parsed" || infer_clicked.value == "raw"){
    functionType = "infer"
    dataType = infer_clicked.value
    text_status.value = 'Select data for classification';
}
if (infer_clicked.value == "parsed" || infer_clicked.value == "raw" || train_clicked.value == "parsed" || train_clicked.value == "raw")
{
    let input = document.getElementById('inputOS')
    train_clicked.value = null
    text_status.title='Status:'; 
    text_status.visible = true;

    file_contents = [];
    file_names = [];

    function error_handler(evt) {
        console.log("In error_handler()")
        if(evt.target.error.name === "NotReadableError") {
          alert("Can't read file!");
        }
    }

    function all_Loaded(){
        // Change infer,train datasource only when all parsed files have loaded
        if(file_contents.length == input.files.length){
            let encodedFile = {'file_contents' : file_contents, 'file_name': file_names};
            console.log(encodedFile)
            // To enable input.onchange event fire even with the same file.
            input.value = ''
            switch(functionType) {
                case 'train':
                    text_status.value = "Running Lexicon Extraction...";
                    train_src.data = encodedFile;
                    train_src.trigger("change");
                break;
                case 'infer':
                    text_status.value = "Running Sentiment Classification...";
                    infer_src.data = encodedFile;
                    infer_src.trigger("change");
                    break;
                default:
            }
        }
    }

    function read_file(files) {
        
        Array.from(files).forEach(function(file){
            let reader = new FileReader();
            reader.onload = function(e){
                let b64string = e.target.result;
                file_contents.push(b64string);
                if (dataType == "raw"){
                    file_names.push(file.name);
                }
                else{
                    file_names.push(file.webkitRelativePath);
                }
            }
            reader.onerror = error_handler;

            reader.onloadend = all_Loaded;

            // readAsDataURL represents the file's data as a base64 encoded string
            reader.readAsDataURL(file);
        });
    }

    /*let input = document.createElement('input');
    input.setAttribute('type', 'file');*/
    input.setAttribute('accept','.csv');
    if (dataType == "parsed"){
        input.setAttribute('webkitdirectory','true');
        input.setAttribute('multiple','')
    }
    else{
        input.removeAttribute('webkitdirectory');
        input.removeAttribute('multiple')
    }
    /*input.value = '';
    console.log("Input element created")*/

    //input.addEventListener("change", handleFiles, false);
    
    function handleFiles(){
        console.log("inside Input change event")
        if (window.FileReader) {
            read_file(input.files);
        }
    }

    input.onchange = function(){
        handleFiles();
    }

    /*input.onchange = function(){
        console.log("inside Input change event")
        if (window.FileReader) {
            read_file(input.files);
        }
    };*/

    input.click();
    console.log("input value = ", input.value)
}
