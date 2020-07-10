Rectangle{
    width: 200
    height: 100
    property string artist: "Katy Perry"
    property string song: "Dark Horse" 

    Text{
        text: song + "\n" + artist
        anchor.centerIn: parent
        fontSizeMode: Text.Fit
    }
}