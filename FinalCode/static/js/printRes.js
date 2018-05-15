



$(document).ready(function() {

    var moviestuff= {};
    $('#corr').on('submit', function(event) {
        $.ajax({
            data : {
                indep : $('#indep').val()
            },
            type : 'POST',
            url : '/project_data'
        })
            .done(function(data) {
                if (data.error) {
                    $('#fucked').text(data.error).show();
                    $('#display').hide();
                }
                else {
                    $('#display').html('<img src="data:image/png;base64,' + data + '" />').show();
                    $('#fucked').hide();
                }
            });
            event.preventDefault();
    });


    $('#lin').on('submit', function(event) {
            $.ajax({
            data : {
                budget : $('#namequery').val(),
                genre : $('#genrequery').val(),
                popular : $('#popquery').val(),
                actor : $('#actor').val(),
                releaseD : $('#releaseD').val(),
                vote : $('#votequery').val()
            },
            type : 'POST',
            url : '/linear'
        })
          .done(function(data) {
                if (data.error) {
                    $('#fuckedLin').text(data.error).show();
                    $('#displayLin').hide();
                }
                else {
                    $('#dispAnswer').html(data.answer).show();

                }
            });
            event.preventDefault();
    });


        $('#nonlin').on('submit', function(event) {
            $.ajax({
            data : {
                budget : $('#budg').val(),
                genre : $('#genre').val(),
                popular : $('#popul').val(),
                vote : $('#voteq').val(),
                actor : $('#star').val(),
                releaseD : $('#release').val()
            },
            type : 'POST',
            url : '/nonlinear'
        })
            .done(function(data) {

                if (data.error) {
                    $('#fuckedNonL').text(data.error).show();
                    $('#displayNonL').hide();
                }
                else {
                    $('#dispAns').html(data.nonlinAns).show();

                }
            });
        event.preventDefault();
    });

    $('#checkmv').on('submit', function(event) {
        $.ajax({
            data : {
                title : $('#title').val(),
                genre : $('#genre').val(),
                cast : $('#cast').val(),
                crew : $('#crew').val(),
                runtime: $('#runtime').val(),
                vote_count : $('#vote_count').val(),
                budget : $('#budget').val(),
                revenue : $('#revenue').val(),
                popularity : $('#popularity').val(),
                vote_average : $('#vote_average').val(),
                IMDB : $('#IMDB').val(),
                rotten : $('#rotten').val(),
                metaC : $('#metaC').val(),
                release_date : $('#release_date').val()
            },
            type : 'POST',
            url : '/check'
        })
            .done(function(data) {
                $('#dispCheck').html(data.search).show();
                document.getElementById('submov').style.visibility = 'visible';
                $('#submov').slideDown();
                window.location.hash = '#dispCheck';
                moviestuff = data.movdata


            });
        event.preventDefault();
    });

    $("#submov").click(function() {
        $.ajax({
            data : moviestuff,
            type : 'POST',
            url : '/addMovie'
        })
            .done(function(data) {

                $('#done').html('<p>Success!</p>Here are the latest entries to the database: (Last 30 Days)'+data.win).show();
                window.location.hash = '#done';


            });
        $(this).slideUp();
        event.preventDefault();

    });

    $('#srch').on('submit', function(event) {
        $.ajax({
            data : {
                title : $('#title').val()
            },
            type : 'POST',
            url : '/search'
        })
            .done(function(data) {
                if (data.none) {
                    $('#display').html(data.none).show();
                    $('#display2').html("<h2>Cast</h2><br>"+data.two).hide();
                    $('#display3').html(data.three).hide();
                    $('#display4').html(data.four).hide();

                } else if (data.one) {
                    $('#display').html("<h2>Title and Genre</h2><br>"+data.one).show();
                    $('#display2').html("<h2>Cast</h2><br>"+data.two).show();
                    $('#display3').html(data.three).show();
                    $('#display4').html(data.four).show();

                } else {
                    $('#display').html(data.more).show();
                    $('#display2').html("<h2>Cast</h2><br>"+data.two).hide();
                    $('#display3').html(data.three).hide();
                    $('#display4').html(data.four).hide();

                }
            });
        event.preventDefault();
    });

    $("#randomF").click(function() {
        $.ajax({
            type : 'POST',
            url : '/rForest'
        })
            .done(function(data) {

                $('#randF').html('<p>Mean Error Bad!</p>Using 1000 Trees/Train Test Split and KFolds Cross<br>'+data.rand).show();
                window.location.hash = '#randF';


            });
        event.preventDefault();

    });

    $("#LinError").click(function() {
        $.ajax({
            type : 'POST',
            url : '/LinError'
        })
            .done(function(data) {

                $('#LinErr').html('<p>Check the Percent Error!</p>One positive is that it is able to match with some 0 values<br>'+data.linerr).show();
                window.location.hash = '#LinErr';


            });
        event.preventDefault();

    });



});



