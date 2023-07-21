// Disable autoDiscover to prevent Dropzone from automatically finding and attaching to all elements with the class "dropzone".
Dropzone.autoDiscover = false;

// Initialize Dropzone.
function init() {
    const dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    // Remove the first file if more than one file is added.
    dz.on("addedfile", () => {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    // Handle completed file upload.
    dz.on("complete", (file) => {
        const imageData = file.dataURL;
        const url = "http://127.0.0.1:5000/classify_image";

        // Send the image data to the server to classify the image.
        $.post(url, { image_data: imageData }, (data, status) => {
            console.log(data);

            if (!data || data.length == 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            const players = ["cristiano_ronaldo", "lionel_messi", "neymar"];
            let match = null;
            let bestScore = -1;

            // Find the player with the highest probability score.
            for (let i = 0; i < data.length; ++i) {
                const maxScoreForThisClass = Math.max(...data[i].class_probability);
                if (maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }

            if (match) {
                // Show the result table.
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();

                // Set the result table based on the match.
                $("#resultHolder").html($(`[data-player="${match.class}"]`).html());

                // Set the score for each player.
                const classDictionary = match.class_dictionary;
                for (let personName in classDictionary) {
                    const index = classDictionary[personName];
                    const probabilityScore = match.class_probability[index];
                    const elementName = `#score_${personName}`;
                    $(elementName).html(probabilityScore);
                }
            }

            // Remove the file from Dropzone.
            // dz.removeFile(file);
        });
    });

    // Process the file queue when the submit button is clicked.
    $("#submitBtn").on("click", (e) => {
        dz.processQueue();
    });
}

$(document).ready(() => {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    // Initialize the app.
    init();
});