import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtCharts 2.3
import QtPositioning 5.12
import QtLocation 5.12
import QtQml 2.12

ChartView {
    id: graph
    title: "Cases per Country"
    //antialiasing: true
    theme: ChartView.ChartThemeBlueCerulean

    property var type: ""
    property var country: ""
    property var sta: ""
    property alias series_name: graphData.name
    
    //Animates the grid lines
    animationOptions: {ChartView.GridAxisAnimations}

    //Set the background colors
    //backgroundColor: "#00FFF5"
    backgroundRoundness: 5
    
    //Controls the legend

    // One country
    SplineSeries {
        id: graphData
        name: ""

        axisX: DateTimeAxis {
                    id: xAxis
                    format: "MMM yyyy"
                }

        axisY: ValueAxis{
                    id: yAxis
                    min: 0
                    max: 1000000
                }

        Component.onCompleted:{
            axisY.applyNiceNumbers()
        }
        color: "orange"
    }
    
    Component.onCompleted:{
        if(graph.addNewSeries(country, type) == 0){
            graphData.name = country
            if(type == "countrycases" || type == "countycases"){
                graph.title = "Total Confirmed Cases in " + country
            } 
            else if(type == "countrydeathes" || type == "countydeathes"){
                graph.title = "Total Confirmed Deathes in " + country
            }
            else if(type == "countryrecovered"){
                graph.title = "Total Recovered in " + country
            }
        }
    }
    
    // This functions adds a country or states coronavirus data to the graph
    // This function accepts either "cases" or "deathes" as a graph type
    // This function accepts the name of a country for country

    function addNewSeries(country, type){
        graphData.removePoints(0, graphData.count)
        console.log(graphData.count)

        var data = null
        
        // Set the data to the appropriate times series data as indicated by type
        if (type == "countrycases"){
            data = corona.countryallconfirmed(country)
        }

        else if (type == "countrydeathes"){
            data = corona.countryalldeath(country)            
        }

        else if (type == "countryrecovered"){
            data = corona.countryallrecovered(country)
        }

        else if (type == "countycases"){
            console.log(country)
            console.log(sta)
            data = corona.countyallconfirmed(country, sta)
        }

        else if (type == "countydeathes"){
            data = corona.countyalldeath(country, sta)
        }

        if (data == null){
            return -1
        }

        console.log(data)

        var max = 0
        var i = 0
        var start = 0
        var end = 0
        var date = 0
        // Add each data point to the graph
        for(var key in data){
            var value = data[key]

            if (value > max){
                max = value
            }
            
            date = new Date(key)
            
            if (i == 0){
                start = date
            }

            graphData.append(date, value)
            i++
        }
        end = date

        //Set the sizes of the axes
        yAxis.max = max + 10000
        xAxis.min = start
        xAxis.max = end
        graphData.index
        return 0
    }
}