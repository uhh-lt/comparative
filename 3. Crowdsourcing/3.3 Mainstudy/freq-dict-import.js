var ElasticsearchCSV = require('elasticsearch-csv');
 
// create an instance of the importer with options 
var esCSV = new ElasticsearchCSV({
    es: { index: 'my_index', type: 'my_type', host: '192.168.0.1' },
    csv: { filePath: '/home/foo/bar/mycsv.csv', headers: true }
});
 
esCSV.import()
    .then(function (response) {
        // Elasticsearch response for the bulk insert 
        console.log(response);
    }, function (err) {
        // throw error 
        throw err;
    });