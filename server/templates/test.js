function BrowseFolder(){
    try{
        var Message = "请选择文件夹";  //选择框提示信息
        var Shell = new ActiveXObject( "Shell.Application" );
        var Folder = Shell.BrowseForFolder(0,Message,0x0040,0x11);//起始目录为：我的电脑
        //var Folder = Shell.BrowseForFolder(0,Message,0); //起始目录为：桌面
        if(Folder != null){
            Folder = Folder.items();  // 返回 FolderItems 对象
            Folder = Folder.item();  // 返回 Folderitem 对象
            Folder = Folder.Path;   // 返回路径
            if(Folder.charAt(Folder.length-1) != "\\"){
                Folder = Folder + "\\";
            }
            document.all.savePath.value=Folder;
            return Folder;
        }
    }catch(e){
        alert(e.message);
    }
}