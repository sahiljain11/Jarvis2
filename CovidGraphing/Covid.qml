import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtCharts 2.3
import QtPositioning 5.9
import QtLocation 5.9
import QtQml 2.12
import "../components"
import "."


//ApplicationWindow{
//visibility: "Maximized"
JarvisWidget{
    id: wid
    width: 700*1.67
    height: 700

    signal graphChanged()
    Rectangle{
        width:  parent.width
        height: parent.height
        color: "#00fff5"
        opacity: 0.5

        Picture{
            image: "../images/frame1.png"
            anchors.fill: parent
        }
    }

    // Placeholder for coronavirus news/symptoms bar 
    Text{
        id: info
        //Position coronavirus news bar at the top of widget
        anchors{
            top: parent.top
            topMargin: parent.height/60
            left: parent.left
            leftMargin: parent.width/40
            right: parent.right
            rightMargin: parent.width/40
        }
        text: "COVID-19 Tracker"
        font.pixelSize: 20
        horizontalAlignment: Text.AlignHCenter
        height: parent.height/12
        color:"white"
    }

    // Shows the corona graphs
    ListView{
        id: root
        //Set visual parameters
        snapMode: ListView.SnapOneItem
        highlightRangeMode: ListView.StrictlyEnforceRange
        highlightMoveDuration: 250
        orientation: ListView.Horizontal
        boundsBehavior: Flickable.StopAtBounds
        clip: true

        anchors{
            right: parent.right
            rightMargin: parent.width/40
            left: parent.horizontalCenter
            leftMargin: parent.width/20
            top: info.bottom
            topMargin: parent.height/40
            bottom: chart_search.top
            bottomMargin: parent.height/30
        }

        model: corona_graphs

        delegate: Component {
            SplineGraph{
                id: cases_per_country
                title: "Cases per Country"
                antialiasing: true
                country: coun
                type: graph_type
                sta: stat
                width: root.width
                height: root.height
            }
        }
    }

    ListModel{
        id: corona_graphs
    }

    // Map plugin for the corona map
    Plugin{
        id: mapPlugin
        name: "esri"

        PluginParameter{
            name: "mapbox.access_token"
            value: "sk.eyJ1IjoiY29ucmFkbGlzdGUiLCJhIjoiY2tkbDk4NGpnMDd6dTJ0cGVib2x2YTMyeiJ9.Ils_duGMNfZ551PW3FDn4w"
        }

        Component.onCompleted: {
            console.log(availableServiceProviders)
        }
    }


    // Corona Virus map
    Map {
        id: map

        // Positions the coronavirus map
        anchors{
            right: root.left
            rightMargin: parent.width/40
            left: parent.left
            leftMargin: parent.width/40
            top: info.bottom
            topMargin: parent.height/40
            bottom: dangerInfo.top
            bottomMargin: parent.height/30
        }

        minimumZoomLevel: 3
        plugin: mapPlugin
        activeMapType: supportedMapTypes[0]

        //Displays information about he geographical location
        Label {
            id: covidInfo
            anchors{
                right: map.right
                rightMargin: map.width/20
                bottom: map.bottom
                bottomMargin: map.height/30
            }
            text: ""
            verticalAlignment: Text.AlignBottom
            width: 100
            height: 100
        }

        Component.onCompleted:{
            parent.addCountryMapItems()
            //parent.addStateMapItems()
        }
    }

    // Handles the finding the locations for each address query
    GeocodeModel {
        id: geoModel
        plugin: map.plugin
        limit: 1

        onLocationsChanged: {
            console.log(count)
            if(count == 1){
                map.zoomLevel = 4
                map.center.latitude = get(0).coordinate.latitude
                map.center.longitude = get(0).coordinate.longitude
            }
        }

        Component.onCompleted:{
            geoModel.query = "United States"
            geoModel.update()
        }
    }

    // Defines the circle representing the number of active cases in an area
    Component {
        id: del
        MapCircle {
            id: point
            radius: 100000
            color: "red"
            border.color: "dark red"
            border.width: 2
            smooth: true
            opacity: 0.5
            center: locationData.coordinate
        }
    }


    // Handles placing the map items
    GeocodeModel {
        id: mapItems
        plugin: mapPlugin
        limit: 1

        Component.onCompleted:{
            mapItems.query = "Russia"
            mapItems.update()
        }
    }

    // Finds the location of the 
    function findLocation (addr) {
        geoModel.query = addr
        geoModel.update()
    }
    

    //Adds the countries to the map
    function addCountryMapItems(){
        map.clearMapItems()
        var num_countries = corona.get_num_countries()
        var countries = corona.get_countries()
        var max_active = 0
        var active_cases = []

        //Loop once through the countries to find the max cases and normalize active cases
        for (var i = 0; i < num_countries; i++){
            //Grab information about number of cases for each country
            var country = countries[i]
            var name = country[0]
            var cases = corona.confirmglobal(name)
            var deathes = corona.deathglobal(name)
            var recovered = corona.recoverglobal(name)
            var active = cases - deathes - recovered
            
            //Append new active caes to the list
            active_cases.push(active)

            //Change the max if a new max is found
            if (max_active < active){
                max_active = active
            }
        }

        // Loop through the active cases, normalize them by the max, and add the countries to the map
        for(var i = 0; i < num_countries; i++){

            //Normaliez active cases by the max_active cases
            var norm_active = active_cases[i]/max_active

            // Add the circle to the map
            var circle = Qt.createQmlObject('import QtLocation 5.12; MapCircle {color: "red"; border.color: "dark red"; border.width: 2; smooth: true; opacity: 0.5}', wid)
            circle.center = QtPositioning.coordinate(countries[i][1], countries[i][2])
            circle.radius = 20000 + (1000000-20000)*norm_active
            map.addMapItem(circle)
        }
    }

    function addStateMapItems(){
        map.clearMapItems()
        var num_states = 50 
        var states = corona.get_states()
        var max_active = 0
        var cases_per_state = []

        //Loop once through the countries to find the max cases and normalize active cases
        for (var i = 0; i < num_states; i++){
            //Grab information about number of cases for each country
            var state = states[i]
            var state_dict = corona.get_data_for_state(state)
            var active = state_dict['Active'][state]

            //Append new active caes to the list
            cases_per_state.push(active)
            console.log(active)

            //Change the max if a new max is found
            if (max_active < active){
                max_active = active
            }
        }

        // Loop through the active cases, normalize them by the max, and add the countries to the map
        for(var i = 0; i < num_states; i++){

            //Normaliez active cases by the max_active cases
            var norm_active = cases_per_state[i]/max_active
            var state_dict = corona.get_data_for_state(states[i])
            // Add the circle to the map
            var circle = Qt.createQmlObject('import QtLocation 5.12; MapCircle {color: "red"; border.color: "dark red"; border.width: 2; smooth: true; opacity: 0.5}', wid)
            circle.center = QtPositioning.coordinate(state_dict['Lat'][states[i]], state_dict['Long_'][states[i]])
            circle.radius = 50000 + (100000-50000)*norm_active
            map.addMapItem(circle)
        }
    }


    //Searchbar for graph button
    TextField{
        id: chart_search
        anchors{
            horizontalCenter: root.horizontalCenter
            bottom: parent.bottom
            bottomMargin: parent.height/20
        }
        
        width: parent.width/8
        height: parent.height/15

        Keys.onPressed:{
            if(event.key == Qt.Key_Return){

                //Case for searching for a country
                if (toggle.word == "Search Countries"){
                    //Replace United States with US to match data
                    if (chart_search.text == "United States"){
                        chart_search.text = "US"
                    }

                    //Replace Taiwan with Taiwant* to match data
                    if (chart_search.text == "Taiwan") {
                        chart_search.text = "Taiwan*"
                    }

                    var country = corona.auto_correct_country_query(chart_search.text)
                    
                    //Clear the graphs
                    corona_graphs.clear()

                    //Return early if there is no match
                    if(corona.get_data_for_country(country) == null){
                        return
                    }

                    //Set new covid graphs for the query                    
                    corona_graphs.append({coun: country, stat: "", graph_type: "countrycases"})
                    corona_graphs.append({coun: country, stat: "", graph_type: "countrydeathes"})
                    corona_graphs.append({coun: country, stat: "", graph_type: "countryrecovered"})

                    //Set the words
                    var cases = corona.confirmglobal(country)
                    var deathes = corona.deathglobal(country)
                    var recovered = corona.recoverglobal(country)
                    var active = cases - deathes - recovered
                    covidInfo.text = "              " + country + "\nTotal Cases: " + Number(cases).toLocaleString(Qt.locale("en"),'f', 0) + "\nTotal Deathes: " + Number(deathes).toLocaleString(Qt.locale("en"),'f', 0) + "\nActive Cases: " + Number(active).toLocaleString(Qt.locale("en"),'f', 0)
                    
                   
                    // Query for the location and center the camera in it
                    parent.findLocation(country)
                }
                
                //Case for searching for counties
                else if (toggle.word == "Search Counties") {
                    var query = chart_search.text

                    // Remove the word "county" from the query
                    query = query.replace(" County", "")
                    query = query.replace(" county", "")

                    var i = 0

                    //Find the comma's position in the query to use as a delimiter
                    while(i < query.length-1 && query.charAt(i) != ","){
                        i++
                    }
                    
                    //return since the query does not contain a comma
                    if (i == query.length-1){
                        return
                    }

                    //Parse the county and state out
                    var county = query.substring(0, i).toString()
                    var state = query.substring(i+1, query.length).toString()

                    //Remove the space if it is at the beginning
                    if(state.charAt(0) == " "){
                        state = state.substring(1, state.length).toString()
                    }

                    //Auto correct the state and the county if need be
                    county = corona.auto_correct_county_query(county)
                    state = corona.auto_correct_state_query(state)
                    
                    //Clear the graphs
                    corona_graphs.clear()

                    //Return early if there is no match
                    if(corona.get_data_for_county(county, state) == null){
                        return
                    }

                    corona_graphs.append({coun: county, stat: state, graph_type: "countycases"})
                    corona_graphs.append({coun: county, stat: state, graph_type: "countydeathes"})

                    var state_dict = corona.get_data_for_state(state)
                    var cases = state_dict['Confirmed'][state]
                    console.log(cases)
                    var deathes = state_dict['Deaths'][state]
                    var active = state_dict['Active'][state]
                    covidInfo.text = "              " + state + "\nTotal Cases: " + Number(cases).toLocaleString(Qt.locale("en"),'f', 0) + "\nTotal Deathes: " + Number(deathes).toLocaleString(Qt.locale("en"),'f', 0) + "\nActive Cases: " + Number(active).toLocaleString(Qt.locale("en"),'f', 0)                   

                    
                    // Query for the location and center the camera in it
                    parent.findLocation(county + " County, " + state)
                }



            }
        }
    }

    //Toggle between searching for county or country
    Switch{
        id: toggle

        //Positioning
        anchors{
            left: chart_search.left
            right: chart_search.right
            bottom: chart_search.top
        }
        height: 20

        //Set the default parameters
        word: "Search Countries"
        iconOff: "../images/back2.png"
        iconOn: "../images/back2.png"

        //Toggle functionality
        onTouched:{
            if (toggle.state == "ON"){
                toggle.word = "Search Counties"
                parent.addStateMapItems()
            }
            else{
                toggle.word = "Search Countries"
                parent.addCountryMapItems()
            }
        }
    }

    //Placeholder for map button
    Label{
        id: dangerInfo
        
        anchors{
            bottom: parent.bottom
            bottomMargin: parent.height/20
        }
        text: ""
        width: parent.width/8
        height: parent.height/20
    }
}

