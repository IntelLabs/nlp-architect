//# ******************************************************************************
//# Copyright 2017-2018 Intel Corporation
//#
//# Licensed under the Apache License, Version 2.0 (the "License");
//# you may not use this file except in compliance with the License.
//# You may obtain a copy of the License at
//#
//#     http://www.apache.org/licenses/LICENSE-2.0
//#
//# Unless required by applicable law or agreed to in writing, software
//# distributed under the License is distributed on an "AS IS" BASIS,
//# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//# See the License for the specific language governing permissions and
//# limitations under the License.
//# ******************************************************************************

const api = window.location.href.slice(0,-10); // slice the url without the "/demo.html" ending

function doAnnotate(text) {
    var displacy_div = document.getElementById("displacy_parse");
    while (displacy_div.firstChild) {
        displacy_div.removeChild(displacy_div.firstChild);
    }
    var res_data = postData(text, api);
}

function getRandomColor(){
    var o = Math.round, r = Math.random, s = 255;
    var r_1 = o(r()*s);
    var r_2 = o(r()*s);
    var r_3 = o(r()*s);
    var rgb = 'rgb(' + r_1 + ',' + r_2 + ',' + r_3 + ')';
    var rgba = 'rgba(' + r_1 + ',' + r_2 + ',' + r_3 + ', 0.2)';
    return [rgb, rgba];
}

function addColorToAnnotationSet(annotation_set){
    dataLength = annotation_set.length
    for (var i = 0; i < dataLength; i++) {
        var colors = getRandomColor()
        var rgb_color = colors[0]
        var rgba_color = colors[1]
        var data1 = '[data-entity][data-entity=' + annotation_set[i] + '] {background: ' + rgba_color +'; border-color: ' + rgb_color + ';}'
        var data2 ='[data-entity][data-entity=' + annotation_set[i] + ']::after {background: ' + rgb_color + ';}'
        document.styleSheets[0].insertRule(data1);
        document.styleSheets[0].insertRule(data2);
    }
}

function renderData(data, type){
    if (type == 'core'){
    //    for dep

    var dataLength = data.length;
    for (var i = 0; i < dataLength; i++) {
        var displacy_div = document.getElementById("displacy_parse");
        var child_node = document.createElement("div");        // Create a child div node
        var div_id = "displacy_div" + i;
        child_node.setAttribute("id", div_id);  // set id
        displacy_div.appendChild(child_node);  // add to html the new div
        displacy_div.appendChild(document.createElement("br"));

        var displacy_core = new displaCy(api,{
            container: '#' + div_id,
            format: 'spacy',
            distance: 150,
            offsetX: 100,
            collapsePunct: false,
            collapsePhrase: false,
            bg: '#006680',
            color: '#000000',
            wordSpacing: 50
        });
        displacy_core.render(data[i]);
        }
    }
    else if (type == 'high_level'){
        var displacy_annotate = new displaCyENT(api,{
        container: '#displacy_models',
        format: 'spacy',
        distance: 300,
        offsetX: 100
        });
        addColorToAnnotationSet(data['annotation_set'])
        displacy_annotate.render(data['doc_text'], data['spans'], data['annotation_set']);
    }
    else{
        console.log("Error, bad response from server - no service type selected")
        window.alert("Error, bad response from server - no service type selected");
    }
}

function postData(text, url) {
  fetch(url, {
    body: JSON.stringify({"docs" : [{"id": 1, "doc": text}]}), // must match 'Content-Type' header
    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
    credentials: 'same-origin', // include, *omit
    headers: {
      'Access-Control-Allow-Origin': '*',
      'content-type': 'application/json',
      'Access-Control-Allow-Headers': '*',
      'Access-Control-Allow-Methods': '*',
      'user-agent': 'Mozilla/4.0 MDN Example',
      'Response-Format': 'json',
      'IS-HTML': 'True'

    },
    method: 'POST', // *GET, PUT, DELETE, etc.
    mode: 'cors', // no-cors, *same-origin
    redirect: 'follow', // *manual, error
    referrer: 'no-referrer', // *client
  })
  .then(response => response.json())
  .then(data => {
    var docData = data
    console.log(docData)
    renderData(docData[0]['doc'], docData[0]['type'])
  })
  .catch(error => {
     console.error(error)
  })
}