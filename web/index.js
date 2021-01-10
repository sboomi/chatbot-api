var chatMessages = [];

let btn = document.getElementById('sendButton')

btn.addEventListener("click", function() {
    let msg = document.getElementById("userMessage").value
    var xhr = new XMLHttpRequest();
    // xhr.setRequestHeader("Content-Type", "application/json; charset=UTF-8");
    document.getElementById("testUserMsg").innerHTML = msg;
    
    xhr.open('POST', 'http://localhost:5000/message');
    let test = {"data": msg}
    // send the request with the song
    let dataString = JSON.stringify(test);
    xhr.timeout = 10000;

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
        var jsonData = JSON.parse(xhr.responseText);
        if (xhr.status === 200) {
            console.log('successful');
            let response = jsonData.response;
            document.getElementById("testBotMsg").innerHTML = response;  
            } else {
            console.log('failed');
            document.getElementById("testBotMsg").innerHTML = "I'm sorry there was a request error";  
            }
        }
    };

    xhr.send(dataString);


    

    //Receive response
    xhr.addEventListener("load", function() {
        let response = "";
        if(xhr.status == 200) {
            let result_api = JSON.parse(xhr.responseText);
            response = result_api.response; 
            document.getElementById("testBotMsg").innerHTML = response;  
        } else {
            response = "No response available"
            alert(`${xhr.status}: ${xhr.statusText}`);
        }
    });

    alert(`Your response is ${response}`);

    document.getElementById("userMessage").value = "";

    
    
})