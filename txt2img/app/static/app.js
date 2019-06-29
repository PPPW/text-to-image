let loc = window.location,
    url = `${loc.protocol}//${loc.hostname}:${loc.port}/gen_img`;

$(function () {        
    let $title = $('#title'),
        $description = $('#description'),
        $imgs = $('#imgs');
       
    let html = '';
    for (let i = 0; i < 9; i++) {
        if (i % 3 === 0) html += '<div class="img-row">';
        html += '<img />';
        if (i % 3 === 2) html += '</div>';
    }
    $imgs.html(html);

    $('#submit').click(function () {
        fade($imgs, 1);
        $.ajax({
            type: 'POST',
            url: url,   
            data: { description: $description.text() },  
            //dataType: 'json',
            success: function (data) {                
                //$imgs.html('<img src="data:image/jpg;base64,' + data + '" />');
                //$imgs.html('<img src="' + data.url + '" />');
                $imgs.find('img').each(function (i) {
                    $(this).attr('src', data.url[i]);
                });
                fade($imgs);
            },
            error: function (jqXHR, textStatus, errorThrown) {
                console.log(errorThrown);
            }
        });
    });
})

function fade($imgs, out) {
    $imgs.find('img').each(function (i) {
        let delay = 2;
        if (i === 4) delay = 0;
        else if (i % 2 === 1) delay = 1;
        if (out) $(this).fadeOut(1000);
        else $(this).fadeIn(1000 + 800 * delay);
    });
}
