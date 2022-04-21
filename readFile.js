var fs = require('fs');
var request = require('request');
  
// Use fs.readFile() method to read the file
// fs.readFile('/home/sanjeev/Downloads/dummy_maths_equtions.pdf', 'utf8', function(err, data){
      
//     // Display the file content
//     if(err){
//         console.log(err.message);
//         console.log("Error in reading!!!!!!!!!!!!");
//     }

//     else{
//         console.log(data);
//         console.log("successfuly opened !!!!!!!!!");
//     }
    
// });

fs.open('/home/sanjeev/Downloads/dummy_maths_equtions.pdf', 'r+', (err, fd) => {

    var buf = new Buffer.alloc(1048576);
    if(err){
        console.log("error in opening eqn.pdf");
    }

    else{
        fs.read(fd, buf, 0, buf.length, 0, function(err, bytes){
            if (err){
               console.log(err);
            }
            
              
            // Print only read bytes to avoid junk.
            if(bytes > 0){
               console.log(buf.slice(0, bytes).toString());
               console.log(bytes + " bytes read");

            //    fs.writeFileSync('eqn.pdf', buf.slice(0, bytes), function(err) {
            //     if (err) {
            //        return console.error(err);
            //     }
                  
            //     console.log("Data written successfully!");
            //  });


            request.post(
                'http://0.0.0.0:5001/',
                { json : {pdf : buf.slice(0, bytes)}},
                function (error, response, body) {
                    if (!error && response.statusCode == 200) {
                        console.log(body);
                    }
                }
            );
            }
         });


    }
})
  
console.log('readFile called');